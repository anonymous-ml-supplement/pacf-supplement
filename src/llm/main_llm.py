#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM instruction tuning / evaluation pipeline for PACF / LoRA / Flat-LoRA / Full FT.

This is a single-file pipeline used for the anonymous PACF supplemental artifact.
No user-identifying information is included.

Key modes:
  - single: run one configuration
  - auto/A/C: run stage A/B/C style grids (see README)
  - summary: summarize scored outputs

Outputs: use `--out` to control the run folder (recommended: `runs/llm/<exp_name>`).
"""

# ================================================================
#  LLM adaptation utilities for LoRA, Flat-LoRA, and PACF-Cons.
#  The script supports controlled training and evaluation for math, chat, and code tasks.
# ================================================================

import os, math, json, random, time, argparse, logging, re
import numpy as np
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

# FlashAttention2
try:
    from transformers.utils import is_flash_attn_2_available
    FLASH2 = is_flash_attn_2_available()
except:
    FLASH2 = False

print(f"[ATTN] flash_attention_2 = {FLASH2}")
import datasets
datasets.disable_progress_bar()
import csv
import gc

# ============================================================
# Utils
# ============================================================

def ts():
    return time.strftime("%Y%m%d-%H%M%S")

def mkdir(x):
    os.makedirs(x, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # ← 這裡修正
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def total_steps_estimate(n, bs, epochs, accum=1):
    return int((n / bs / accum) * epochs)

def log_run_config(out, task, variant, mode, run_name, args):
    mkdir(os.path.join(out, "run_configs"))
    path = os.path.join(out, "run_configs", f"{run_name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "task": task,
                "variant": variant,
                "mode": mode,
                "run_name": run_name,
                "args": args,
            },
            f,
            indent=2,
        )
    print(f"[LOG] config → {path}")

def log_eval_row(out, task, variant, mode, run_name, args,
                 metric_name, metric_value, n_eval):
    """
    Append one evaluation record into out/eval_cache/eval_cache.csv
    using csv.writer, so JSON args (with commas) 
    """
    cache = os.path.join(out, "eval_cache")
    mkdir(cache)
    path = os.path.join(cache, "eval_cache.csv")
    header = ["ts", "task", "variant", "mode", "run_name", "metric", "value", "n", "args"]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            ts(),
            task,
            variant,
            mode,
            run_name,
            metric_name or "",
            metric_value if metric_value is not None else "",
            n_eval,
            json.dumps(args),
        ])
    print("[LOG] eval_cache updated.")

def summarize_eval_cache(out):
    import pandas as pd

    path = os.path.join(out, "eval_cache", "eval_cache.csv")
    if not os.path.exists(path):
        print("[WARN] eval_cache missing")
        return
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    g = df.groupby(["task", "variant"])["value"].agg(["mean", "std", "count"])
    print(g)
    mkdir(os.path.join(out, "summary"))
    g.to_csv(os.path.join(out, "summary", "summary.csv"))
    print("[SAVE] summary complete")

# ============================================================
# Data
# ============================================================

MATH_SUFFIX = "\n\nSolve step by step and give the final answer in the format: #### <number>"
MATH_PROMPT = "Question:\n{q}\n\nAnswer:"
CHAT_PROMPT = "Instruction:\n{q}\n\nResponse:"

TEMPLATE_WO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

ALPACA_PREFIX_TEMPLATE_MD = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Complete the following Python code.\n"
    "Notes:\n"
    "- Respond with the entire complete function definition.\n"
    "- Do not add any comments.\n"
    "- Be as concise in your code as possible.\n"
    "- Use only built-in libraries, assume no additional imports other than those provided (if any).\n"
    "- Use four spaces for each level of indentation.\n\n"
    "Code:\n"
    "{PROMPT}\n\n"
    "### Response:\n"
)


def post_process_humaneval_completion(text: str) -> str:
    """
    Post-process model output for HumanEval so that it contains only a
    valid Python function definition.

    """
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("\t", "    ")

    lines = [l.rstrip() for l in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    start = 0
    for i, line in enumerate(lines):
        if "def " in line:
            start = i
            break
    lines = lines[start:]

    min_spaces = None
    for line in lines:
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        if min_spaces is None or leading < min_spaces:
            min_spaces = leading
    if min_spaces is None:
        min_spaces = 0

    trimmed = [line[min_spaces:] if len(line) >= min_spaces else line for line in lines]
    return "\n".join(trimmed) + "\n"


# -------- Flat-style MetaMath (GSM subset, length-filtered) --------

def load_metamath_100k_flat(tokenizer, max_tokens=512, n_train=100000):
    """
    MetaMath loader aligned with Flat-LoRA official code:

    - dataset: meta-math/MetaMathQA
    - filter: only keep samples whose `type` contains "GSM" (if column exists)
    - x = "Q: {query}\\nA: "
    - y = response split at "\\nThe answer is:" 
    - length filter: len(tokenizer(x + " " + y)) < max_tokens
    - take first n_train (=100000) valid samples
    """
    print("[INFO] Loading MetaMathQA (Flat-style, aligned with Flat-LoRA)")
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    col_names = set(ds.column_names)
    has_type = "type" in col_names
    if not has_type:
        print(
            "[WARN] MetaMathQA has no 'type' column; "
            "falling back to length-filtered first 100k without GSM filtering."
        )

    ds = ds.shuffle(seed=42)

    train_samples = []
    total = 0
    ok = 0

    for ex in ds:
        total += 1

        if has_type:
            t = str(ex.get("type", ""))
            if "GSM" not in t:
                continue

        q = ex.get("query", "").strip()
        resp = ex.get("response", "")
        if not q or not resp:
            continue

        a_reason = resp.split("\nThe answer is:")[0].strip()
        if not a_reason:
            continue

        x = f"Q: {q}\nA: "
        y = a_reason

        ids = tokenizer(x + " " + y, add_special_tokens=False)["input_ids"]
        if len(ids) >= max_tokens:
            continue

        train_samples.append({"x": x, "y": y})
        ok += 1

        if ok % 10000 == 0:
            print(f"[MetaMath] accepted {ok} / seen {total}")

        if ok >= n_train:
            break

    print(f"[INFO] MetaMath GSM filtered: accepted={ok}, total_seen={total}")
    return train_samples


def load_gsm8k_test():
    """
    Flat-LoRA style GSM8K loader for evaluation.

    Each element:
      {
        "x": "Q: {question}\\nA: ",
        "y": answer_text,
      }
    """
    ds = load_dataset("gsm8k", "main", split="test")
    out = []
    for x in ds:
        out.append(
            {
                "x": f'Q: {x["question"]}\nA: ',
                "y": x["answer"],
            }
        )
    return out

from tqdm import tqdm 

def load_wizardlm(tokenizer, max_tokens=1024):
    ds = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split="train")
    ds = ds.shuffle(seed=42)

    train_samples = []
    total = 0
    ok = 0

    bar = tqdm(ds, total=70000, desc="[WizardLM] filtering")
    for sample in bar:
        total += 1
        instr = sample["instruction"]
        out = sample["output"]

        x = TEMPLATE_WO_INPUT.format(instruction=instr)
        y = out

        low = y.lower()
        if "sorry" in low or "as an ai" in low:
            continue

        ids = tokenizer(x + y, add_special_tokens=False)["input_ids"]
        if len(ids) >= max_tokens:
            continue

        ok += 1
        train_samples.append({"x": x, "y": y})

        bar.set_description(f"[WizardLM] ok={ok} / seen={total}")

        if ok >= 52000:
            break

    print(f"[INFO] WizardLM filtered: accepted={ok}, total_seen={total}")
    return train_samples

def load_mtbench():
    ds = load_dataset("philschmid/mt-bench", split="train")
    out = []
    for ex in ds:
        turns = ex["turns"]
        if turns:
            out.append({"q": turns[0], "meta": {"qid": ex["question_id"]}})
    return out

from tqdm import tqdm  # 如果上面 WizardLM 那邊已經有 import 就不用再加

def load_codefeedback(tokenizer, max_tokens=1024):
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    ds = ds.shuffle(seed=42)

    train_samples = []
    total = 0
    ok = 0

    bar = tqdm(ds, total=110000, desc="[CodeFeedback] filtering")
    for sample in bar:
        total += 1

        ans = sample["answer"]
        if "```" not in ans:
            continue

        y = "```".join(ans.split("```")[:2]) + "```"

        x = TEMPLATE_WO_INPUT.format(instruction=sample["query"])

        ids = tokenizer(x + " " + y, add_special_tokens=False)["input_ids"]
        if len(ids) >= max_tokens:
            continue

        train_samples.append({"x": x, "y": y})
        ok += 1
        bar.set_description(f"[CodeFeedback] ok={ok} / seen={total}")

        if ok >= 100000:  
            break

    print(f"[INFO] CodeFeedback filtered: accepted={ok}, total_seen={total}")
    return train_samples

def load_humaneval():
    ds = load_dataset("openai_humaneval", split="test")
    return [
        {"task_id": x["task_id"], "prompt": x["prompt"], "sol": x["canonical_solution"]}
        for x in ds
    ]

class SFTDataset(Dataset):
    def __init__(self, tokenizer, task, args_global):
        self.tk = tokenizer
        self.data = []

        def build_xy_ids(x, y, max_len):
            tk = self.tk
            x_ids = tk(x, add_special_tokens=False)["input_ids"]
            full_text = x + " " + y
            if tk.eos_token is not None:
                full_text = full_text + tk.eos_token

            full = tk(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )["input_ids"]

            len_x = len(x_ids)
            labels = full.copy()
            for i in range(min(len_x, len(labels))):
                labels[i] = -100  

            return full, labels

        if task == "math":
            raw = load_metamath_100k_flat(self.tk, max_tokens=512, n_train=100000)
            for ex in raw:
                x = ex["x"].strip()  
                y = ex["y"].strip()  
                full_ids, labels = build_xy_ids(x, y, max_len=1024)
                self.data.append({"input_ids": full_ids, "labels": labels})

        elif task == "chat":
            raw = load_wizardlm(self.tk, max_tokens=1024)
            for ex in raw:
                x = ex["x"].strip()          
                y = ex["y"].strip()
                full_ids, labels = build_xy_ids(x, y, max_len=1024)
                self.data.append({"input_ids": full_ids, "labels": labels})

        elif task == "code":
            raw = load_codefeedback(self.tk, max_tokens=1024)
            for ex in raw:
                x = ex["x"].strip()
                y = ex["y"].strip()
                full_ids, labels = build_xy_ids(x, y, max_len=1024)
                self.data.append({"input_ids": full_ids, "labels": labels})

        # ===== Quick debug: take only first N samples =====
        if hasattr(args_global, "debug_n") and args_global.debug_n > 0:
            print(f"[DEBUG] Using only first {args_global.debug_n} samples")
            self.data = self.data[: args_global.debug_n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# ============================================================
# Model: LoRA + Flat-LoRA + PACF
# ============================================================

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r, alpha, dropout):
        super().__init__()
        self.r = r
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_dim))
            self.B = nn.Parameter(torch.zeros(out_dim, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.scaling = alpha / r
            self.drop = nn.Dropout(dropout)
        else:
            self.A = None
            self.B = None

    def forward(self, x, base_weight):
        """
        x: (batch, ..., in_dim)
        base_weight: (out_dim, in_dim)
        """
        if self.r <= 0:
            return torch.matmul(x, base_weight.T)

        # base projection
        out = torch.matmul(x, base_weight.T)

        A = self.A
        B = self.B
        
        assert A.device == x.device and B.device == x.device, \
            f"LoRA params on {A.device}/{B.device}, but x on {x.device}. Fix inject_lora_linear()."
        assert A.dtype == x.dtype and B.dtype == x.dtype, \
            f"LoRA dtype {A.dtype}/{B.dtype}, but x dtype {x.dtype}. Fix new.lora.to(dtype=...)."
        if A.dtype != x.dtype:
            A = A.to(dtype=x.dtype)
            B = B.to(dtype=x.dtype)

        update = torch.matmul(self.drop(x), A.T)
        update = torch.matmul(update, B.T) * self.scaling
        return out + update

def inject_lora_linear(module, r, alpha, dropout, target_modules=None, prefix=""):

    for name, m in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(m, nn.Linear):
            use_lora = True
            if target_modules is not None:
                use_lora = any(t in full_name for t in target_modules)

            if not use_lora:
                continue

            w = m.weight
            if getattr(w, "is_meta", False):
                print(f"[LoRA] skip meta Linear: {full_name} (weight is meta)")
                continue

            new = nn.Linear(m.in_features, m.out_features, bias=(m.bias is not None))
            new.to(device=w.device, dtype=w.dtype)

            with torch.no_grad():
                new.weight.copy_(w.detach())
                if m.bias is not None and new.bias is not None:
                    new.bias.copy_(m.bias.detach().to(new.bias.dtype))

            new.lora = LoRALayer(m.in_features, m.out_features, r, alpha, dropout)
            new.lora.to(device=w.device, dtype=w.dtype)  
            module._modules[name] = new
        else:
            inject_lora_linear(m, r, alpha, dropout, target_modules, prefix=full_name)




def mark_only_lora_trainable(model):
    """
    Freeze base model. Only train LoRA parameters.
    This is REQUIRED for LoRA / Flat-LoRA / PACF to work correctly.
    """
    for name, p in model.named_parameters():
        if "lora" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

# ---- PATCH LINEAR.FORWARD (avoid recursion) ----
_ORIG_LINEAR_FORWARD = nn.Linear.forward  # 保存原本的 forward

def lora_forward(self, x):
    if hasattr(self, "lora"):
        return self.lora(x, self.weight)
    return _ORIG_LINEAR_FORWARD(self, x)

def patch_linear_forward():
    nn.Linear.forward = lora_forward

patch_linear_forward()

def build_lora_model(model_name, lora_r, lora_alpha, lora_dropout):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")

    llama_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    inject_lora_linear(model, lora_r, lora_alpha, lora_dropout, target_modules=llama_targets)

    mark_only_lora_trainable(model)
    return model

# ===== FULL fine-tune (no LoRA injection) =====
def build_full_model(model_name):
    print("[MODEL] build FULL model (no LoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")
    return model


# ------- LoRA adapter save / load (small file) ---------

def get_lora_state_dict(model):
    """
    Extract only LoRA-related parameters from the full state_dict.
    We keep keys that contain 'lora.' (our injected LoRALayer weights).
    """
    full_sd = model.state_dict()
    lora_sd = {k: v.cpu() for k, v in full_sd.items() if "lora." in k}
    return lora_sd

def save_lora_adapter(model, save_dir):
    """
    Save only LoRA parameters to save_dir/lora_adapter.pt
    (plus a tiny adapter_config.json for bookkeeping).
    """
    mkdir(save_dir)
    lora_sd = get_lora_state_dict(model)
    out_path = os.path.join(save_dir, "lora_adapter.pt")
    torch.save(lora_sd, out_path)

    cfg = {
        "format": "custom_lora_v15",
        "n_params": len(lora_sd),
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[SAVE] LoRA adapter (only) → {out_path}")

def load_lora_adapter(model, adapter_path):
    """
    Load LoRA-only checkpoint (lora_adapter.pt) into a fresh base model.
    """
    lora_sd = torch.load(adapter_path, map_location="cpu")
    full_sd = model.state_dict()
    full_sd.update(lora_sd)
    model.load_state_dict(full_sd)
    return model

# ------- Flat-LoRA Gaussian perturbation ---------
def stable_step_seed(base_seed: int, step: int) -> int:
    """
    Deterministic seed = f(base_seed, step), independent of time / python hash.
    """
    base_seed = int(base_seed) & 0x7FFFFFFF
    step = int(step) & 0x7FFFFFFF
    return int((base_seed * 1000003 + step * 9176 + 1337) % (2**31 - 1))
class FlatLoRACallback(TrainerCallback):
    """
    Flat-LoRA official-style implementation (timing aligned),
    but with:
      - deterministic per-block seed (no time.time())
      - noise sampling isolated via torch.random.fork_rng (won't pollute dropout/data RNG)
    """

    def __init__(self, cnt, rho=0.05, T=1, grad_accum_steps=1, base_seed=42):
        super().__init__()
        self.cnt = max(1, int(cnt))
        self.rho = float(rho)
        self.T = int(T)
        self.grad_accum_steps = int(max(1, grad_accum_steps))

        self.grad_counter = 0
        self.base_seed = int(base_seed)
        self.seed = None
        self.filter_norms = []

    def _rng_ctx(self, model):
        dev = next(model.parameters()).device
        if dev.type == "cuda" and dev.index is not None:
            return torch.random.fork_rng(devices=[dev.index])
        return torch.random.fork_rng(devices=[])

    def on_step_begin(self, args, state, control, **kwargs):
        self.grad_counter += 1
        if self.rho <= 0:
            return

        # only at block start
        if (self.grad_counter - 1) % (self.T * self.grad_accum_steps) != 0:
            return

        model = kwargs["model"]

        # cosine factor in [0,1]
        x = min(max(self.grad_counter / self.cnt, 0.0), 1.0)
        factor = 0.5 * (1.0 - math.cos(math.pi * x))

        # deterministic seed per noise block
        block_id = (self.grad_counter - 1) // (self.T * self.grad_accum_steps)
        self.seed = stable_step_seed(self.base_seed, block_id)

        self.filter_norms = []

        with self._rng_ctx(model):
            torch.manual_seed(self.seed)

            for module in model.modules():
                if not hasattr(module, "lora"):
                    continue
                lora = module.lora
                if not isinstance(lora, LoRALayer) or lora.r <= 0:
                    continue

                W = module.weight
                A = lora.A
                B = lora.B

                with torch.no_grad():
                    eff = W.data + lora.scaling * (B @ A)
                    d = eff.shape[1]

                    filter_norm = (
                        factor
                        * (self.rho + 1e-16)
                        / math.sqrt(float(d))
                        * eff.norm(dim=1, keepdim=True)
                    )
                    filter_norm = torch.nan_to_num(filter_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    filter_norm = torch.clamp(filter_norm, min=0.0)
                    self.filter_norms.append(filter_norm)

                    std = filter_norm.repeat(1, d).view_as(W)
                    std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
                    std = torch.clamp(std, min=0.0)

                    # sample noise without touching global RNG outside fork_rng
                    noise = (torch.randn_like(W) * std).to(dtype=W.dtype)
                    W.data.add_(noise)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.rho <= 0 or not self.filter_norms or self.seed is None:
            return

        model = kwargs["model"]

        with self._rng_ctx(model):
            torch.manual_seed(self.seed)

            idx = 0
            for module in model.modules():
                if not hasattr(module, "lora"):
                    continue
                lora = module.lora
                if not isinstance(lora, LoRALayer) or lora.r <= 0:
                    continue

                W = module.weight
                d = W.shape[1]

                filter_norm = self.filter_norms[idx]
                filter_norm = torch.nan_to_num(filter_norm, nan=0.0, posinf=0.0, neginf=0.0)
                filter_norm = torch.clamp(filter_norm, min=0.0)

                std = filter_norm.repeat(1, d).view_as(W)
                std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
                std = torch.clamp(std, min=0.0)

                noise = (torch.randn_like(W) * std).to(dtype=W.dtype)
                W.data.sub_(noise)

                idx += 1


# ============================================================
# Trainer Builder (pass seed in, and feed it to TrainingArguments + FlatLoRA)
# ============================================================


# ------- PACF ---------

class PACFTrainer(Trainer):
    """
    PACF for LoRA :
    """

    def _get_lora_params(self, model):
        params = getattr(self, "_cached_lora_params", None)
        if params is not None:
            return params
        params = []
        for name, p in model.named_parameters():
            if "lora" in name and p.requires_grad:
                params.append((name, p))
        self._cached_lora_params = params
        return params

    def _select_valid_positions(self, logits, labels):
        if logits is None:
            return None

        if labels is None:
            return logits.view(-1, logits.size(-1))

        if labels.dim() != 2:
            labels = labels.view(logits.size(0), -1)

        mask = labels.ne(-100)  # [B, T]
        if mask.sum() == 0:
            return logits.view(-1, logits.size(-1))

        logits = logits[mask]
        return logits

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        lm_loss = loss

        if not getattr(self, "use_pacf", False):
            return (lm_loss, outputs) if return_outputs else lm_loss

        lora_params = self._get_lora_params(model)
        if len(lora_params) == 0:
            return (lm_loss, outputs) if return_outputs else lm_loss

        flat_sigma = float(getattr(self, "flat_sigma", 0.05))
        pac_lambda = float(getattr(self, "pac_lambda", 0.0))
        pac_reg_warmup = float(getattr(self, "pac_reg_warmup", 0.3))
        #total_steps = int(getattr(self, "total_steps", 1))
        #step = int(getattr(self.state, "global_step", 0))
        T = int(getattr(self.state, "max_steps", getattr(self, "total_steps", 1)))
        step = int(getattr(self.state, "global_step", 0))          


        #with torch.no_grad():
        clean_logits = outputs.logits.detach()
        labels = inputs.get("labels", None)
        clean_flat = self._select_valid_positions(clean_logits, labels)

        backups = []
        for name, p in lora_params:
            backups.append(p.data.clone())
            noise = torch.randn_like(p) * flat_sigma
            p.data.add_(noise)
        inputs_wo_labels = {k: v for k, v in inputs.items() if k != "labels"}
        try:
            #with torch.no_grad():
            pert_outputs = model(**inputs_wo_labels)
            pert_logits = pert_outputs.logits
        finally:
            for (name, p), b in zip(lora_params, backups):
                p.data.copy_(b)

        pert_flat = self._select_valid_positions(pert_logits, labels)

        if clean_flat is None or pert_flat is None:
            pac_reg = torch.tensor(0.0, device=lm_loss.device)
        else:
            n = min(clean_flat.size(0), pert_flat.size(0))
            if n == 0:
                pac_reg = torch.tensor(0.0, device=lm_loss.device)
            else:
                clean_flat = clean_flat[:n]
                pert_flat = pert_flat[:n]

                logp_clean = torch.log_softmax(clean_flat.float(), dim=-1)
                logp_pert  = torch.log_softmax(pert_flat.float(),  dim=-1)
                p_clean = logp_clean.exp()
                pac_reg = (p_clean * (logp_clean - logp_pert)).sum(-1).mean()
                pac_reg = pac_reg.to(lm_loss.dtype)


        warm = max(1, int(pac_reg_warmup * T))
        x = min(max(step / warm, 0.0), 1.0)
        warm_factor = 0.5 * (1.0 - math.cos(math.pi * x))


        total_loss = lm_loss + pac_lambda * warm_factor * pac_reg

        if hasattr(self, "log"):
            self.log({
              "pacf_reg": pac_reg.detach(),
              "pacf_weight": pac_lambda * warm_factor,
              "pacf_step": step,
            })


        if return_outputs:
            outputs.pacf_reg = pac_reg.detach()
            outputs.pacf_weight = pac_lambda * warm_factor
            return total_loss, outputs
        return total_loss

# ============================================================
# Evaluation: GSM8K / MT-Bench / HumanEval
# ============================================================

def extract_num(s):
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"####\s*(-?\d+)", s)
    return m.group(1) if m else None


def evaluate_gsm8k_em(model, tokenizer, eval_data, bs=4, max_new_tokens=512):
    """
    Flat-style GSM8K EM:
      - prompt: 'Q: ...\\nA: '
      - greedy (do_sample=False, num_beams=1)
      - parse '#### number'
    """
    model.eval()
    correct = 0
    total = len(eval_data)
    tk = tokenizer
    tk.padding_side = "left"
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token

    for i in range(0, total, bs):
        batch = eval_data[i : i + bs]
        prompts = []
        for ex in batch:
            pt = ex["x"]
            if tk.bos_token:
                pt = tk.bos_token + pt
            prompts.append(pt)

        enc = tk(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tk.eos_token_id,
            pad_token_id=tk.eos_token_id,
        )
        for j in range(len(batch)):
            new = gen[j, enc["input_ids"].shape[1] :]
            pred = tk.decode(new, skip_special_tokens=True).strip()
            gold = batch[j]["y"]
            p = extract_num(pred)
            g = extract_num(gold)
            if p is not None and g is not None and p == g:
                correct += 1

    return correct / total, total

def evaluate_mtbench(model, tk, eval_list, run_name, out_dir, bs=4):
    mkdir(out_dir)
    path = os.path.join(out_dir, f"{run_name}.jsonl")
    fw = open(path, "w")
    tk.padding_side = "left"
    for i in range(0, len(eval_list), bs):
        batch = eval_list[i : i + bs]
        prompts = []
        for ex in batch:
            p = CHAT_PROMPT.format(q=ex["q"])
            if tk.bos_token:
                p = tk.bos_token + p
            prompts.append(p)
        enc = tk(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,     
            num_beams=1,         
            temperature=0.0,
        )
        for j in range(gen.size(0)):
            ans = tk.decode(
                gen[j, enc["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            fw.write(json.dumps({"q": batch[j]["q"], "answer": ans}) + "\n")
    fw.close()
    return path

def evaluate_humaneval_pass1(model, tk, eval_list, num_samples_per_task: int = 5, out_dir: str = "humaneval"):
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    run_name = os.environ.get("PACF_RUN_NAME", "tmp")

    os.makedirs(out_dir, exist_ok=True)
    sample_path = os.path.join(out_dir, f"{run_name}_humaneval_samples.jsonl")

    problems = read_problems()
    tk.padding_side = "left"
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token

    samples = []

    model.eval()
    with torch.no_grad():
        for task_id, problem in problems.items():
            prompt = problem["prompt"]

            prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt)
            if tk.bos_token:
                prompt_in = tk.bos_token + prompt_in

            inp = tk(prompt_in, return_tensors="pt")
            inp = {k: v.to(model.device) for k, v in inp.items()}

            gen = model.generate(
                **inp,
                max_new_tokens=512,
                do_sample=False,     
                num_beams=1,         
                eos_token_id=tk.eos_token_id,
                pad_token_id=tk.eos_token_id,
                temperature=0.0,
            )

            out = tk.decode(
                gen[0, inp["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            completion = post_process_humaneval_completion(out)
            samples.append({"task_id": task_id, "completion": completion})

    write_jsonl(sample_path, samples)
    print(f"[HUMANEVAL] wrote samples → {sample_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = evaluate_functional_correctness(sample_path)
    pass_at_1 = results.get("pass@1", 0.0)
    print(f"[HUMANEVAL] pass@1 = {pass_at_1:.4f}")

    return pass_at_1, len(problems)


# ============================================================
# Trainer Builder
# ============================================================

class LeftPad:
    def __init__(self, tk):
        self.tk = tk
        tk.padding_side = "left"

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad = self.tk.pad_token_id
        ids = []
        labs = []
        att = []
        for ex in batch:
            need = max_len - len(ex["input_ids"])
            ids.append([pad] * need + ex["input_ids"])
            labs.append([-100] * need + ex["labels"])
            att.append([0] * need + [1] * len(ex["input_ids"]))
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att),
        }

def build_trainer(
    model,
    tokenizer,
    train_set,
    variant,
    lr,
    batch_size,
    epochs,
    wd,
    warmup_ratio,
    scheduler,
    grad_accum,
    flat_sigma,
    pac_lambda,
    pac_warm,
    out_dir,
    wandb_project,
    wandb_mode,
    seed,  
):
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        lr_scheduler_type=scheduler,
        save_strategy="no",
        bf16=True,
        report_to=["wandb"] if wandb_project else [],

        seed=int(seed),

        logging_strategy="steps",
        logging_steps=1000,
        disable_tqdm=False,
    )

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_MODE"] = wandb_mode

    coll = LeftPad(tokenizer)
    TrainerCls = PACFTrainer if variant == "lora_pacf" else Trainer
    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_set,
        data_collator=coll,
    )

    total_steps = total_steps_estimate(len(train_set), batch_size, epochs, grad_accum)

    # PACF
    if variant == "lora_pacf":
        trainer.use_pacf = True
        trainer.pac_lambda = pac_lambda
        trainer.pac_reg_warmup = pac_warm
        trainer.total_steps = total_steps
        trainer.flat_sigma = flat_sigma

    # Flat-LoRA
    if variant == "flat_lora":
        n_gpu = max(1, torch.cuda.device_count())
        tot_bz = batch_size * n_gpu
        cnt = math.ceil(len(train_set) / tot_bz) * epochs
        trainer.add_callback(
            FlatLoRACallback(
                cnt=cnt,
                rho=flat_sigma,
                T=1,
                grad_accum_steps=grad_accum,
                base_seed=int(seed), 
            )
        )

    return trainer

# ============================================================
# Stage A / Stage C / Single Run
# ============================================================

class Slice:
    def __init__(self, base, idxs):
        self.base = base
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base[self.idxs[i]]

def run_variant(task, variant, model_name, tk, train, evald, args):
    run_name = (
        f"{task}-{variant}-seed{args.seed}-e{args.epochs}-lr{args.lr}"
        f"-pac{args.pac_lambda}-wr{args.reg_warm}-drop{args.lora_dropout}"
        f"-r{args.lora_r}-a{args.lora_alpha}"
    )
    print(f"[RUN] {run_name}")
    #model = build_lora_model(model_name, args.lora_r, args.lora_alpha, args.lora_dropout)
    if variant == "full":
        model = build_full_model(model_name)
    else:
        model = build_lora_model(model_name, args.lora_r, args.lora_alpha, args.lora_dropout)

    trainer = build_trainer(
        model,
        tk,
        train,
        variant,
        args.lr,
        args.batch,
        args.epochs,
        args.wd,
        args.warmup,
        args.scheduler,
        args.accum,
        args.flat_sigma,
        args.pac_lambda,
        args.reg_warm,
        os.path.join(args.out, "hf_runs", run_name),
        args.wandb_project,
        args.wandb_mode,
        seed=args.seed,   
    )
    trainer.train()
    # save LoRA-only adapter (small file instead of full 7B model)
    #ad = os.path.join(args.out, "adapters", run_name)
    #save_lora_adapter(model, ad)
    # save checkpoint
    if variant == "full":
        full_dir = os.path.join(args.out, "full_models", run_name)
        mkdir(full_dir)
        model.save_pretrained(full_dir)
        tk.save_pretrained(full_dir)
    else:
        ad = os.path.join(args.out, "adapters", run_name)
        save_lora_adapter(model, ad)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # evaluate
    if task == "math":
        em, n = evaluate_gsm8k_em(model, tk, evald, bs=args.eval_bs)
        print(f"[RESULT] GSM8K EM={em}")
        metric = "gsm8k_em"
        val = em
        ne = n
    elif task == "chat":
        p = evaluate_mtbench(
            model,
            tk,
            evald,
            run_name,
            os.path.join(args.out, "mtbench"),
        )
        print(f"[RESULT] MTBench saved {p}")
        metric = "mtbench_raw"
        val = None
        ne = len(evald)
    else:
        p1, n = evaluate_humaneval_pass1(model, tk, evald)
        print(f"[RESULT] HumanEval PASS1={p1}")
        metric = "humaneval_pass1"
        val = p1
        ne = n

    log_run_config(args.out, task, variant, args.mode, run_name, vars(args))
    log_eval_row(args.out, task, variant, args.mode, run_name, vars(args), metric, val, ne)
    return val

def stageA(task, tk, train, evald, args):
    if task != "math":
        return {"pac": args.pac_lambda, "wr": args.reg_warm, "em": None}
    n = len(train)
    k = int(0.25 * n)
    idx = np.random.choice(n, k, replace=False).tolist()
    sub = Slice(train, idx)
    best = {"em": -1}
    for lam in args.grid_lam:
        for wr in args.grid_wr:
            args.pac_lambda = lam
            args.reg_warm = wr
            em = run_variant("math", "lora_pacf", args.model, tk, sub, evald, args)
            if em > best["em"]:
                best = {"pac": lam, "wr": wr, "em": em}
    print("[StageA] BEST=", best)
    return best

def stageC(task, tk, train, evald, args, best=None):
    if best and "pac" in best:
        args.pac_lambda = best["pac"]
        args.reg_warm = best["wr"]
    for v in ["lora", "flat_lora", "lora_pacf"]:
        run_variant(task, v, args.model, tk, train, evald, args)

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["math", "chat", "code"], default="math")
    ap.add_argument("--variant", choices=["lora", "flat_lora", "lora_pacf", "full"], default="lora")
    ap.add_argument("--mode", choices=["single", "auto", "C", "A", "summary"], default="single")
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--out", default="runs_llm")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--flat_sigma", type=float, default=0.05)
    ap.add_argument("--pac_lambda", type=float, default=1e-5)
    ap.add_argument("--reg_warm", type=float, default=0.3)
    ap.add_argument("--grid_lam", type=float, nargs="+", default=[5e-6, 1e-5, 2e-5])
    ap.add_argument("--grid_wr", type=float, nargs="+", default=[0.25, 0.3])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--debug_n", type=int, default=-1, help="use only first N samples for quick sanity check")
    

    
    args = ap.parse_args()
    mkdir(args.out)
    set_seed(args.seed)
    run_name = f"{args.task}-{args.variant}-seed{args.seed}-e{args.epochs}-lr{args.lr}-pac{args.pac_lambda}-wr{args.reg_warm}-drop{args.lora_dropout}-r{args.lora_r}-a{args.lora_alpha}"
    os.environ["PACF_RUN_NAME"] = run_name
    print(f"[RUN] {run_name}")

    
    tk = AutoTokenizer.from_pretrained(args.model)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token

    if args.task == "math":
        train = SFTDataset(tk, "math", args)
        evald = load_gsm8k_test()
    elif args.task == "chat":
        train = SFTDataset(tk, "chat", args)
        evald = load_mtbench()
    else:
        train = SFTDataset(tk, "code", args)
        evald = load_humaneval()

    if args.mode == "summary":
        summarize_eval_cache(args.out)
        return

    if args.mode == "single":
        run_variant(args.task, args.variant, args.model, tk, train, evald, args)
        return

    if args.mode == "auto":
        best = stageA(args.task, tk, train, evald, args)
        stageC(args.task, tk, train, evald, args, best)
        summarize_eval_cache(args.out)
        return

    if args.mode == "C":
        stageC(args.task, tk, train, evald, args, None)
        summarize_eval_cache(args.out)
        return

    if args.mode == "A":
        best = stageA(args.task, tk, train, evald, args)
        print("[StageA only] best =", best)
        return


if __name__ == "__main__":
    main()
