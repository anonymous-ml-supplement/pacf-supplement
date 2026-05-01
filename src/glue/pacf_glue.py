#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLUE (T5) experiments for PACF / LoRA / Flat-LoRA.

This script implements a deterministic Stage A/B/C protocol:
  - Stage A: deterministic grid over (lambda, warmup) candidates
  - Stage B: re-run top-K candidates to confirm
  - Stage C: multi-seed final runs for paper numbers

No user-identifying info is included. Outputs are written under `runs/` by default.

Usage examples (see README):
  python src/glue/pacf_glue.py --mode paper --task sst2 --ranks 16 --seeds_stageC 1,2,3
"""

# Deterministic grid protocol for PACF, LoRA, and Flat-LoRA comparisons.
import os, math, time, random, json, csv, platform, argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    T5ForConditionalGeneration, DataCollatorForSeq2Seq,
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from evaluate import load as _load_metric

# ----------------- helpers -----------------
def _now_tag(): return datetime.now().strftime("%Y%m%d-%H%M%S")
def _csv_exists(p): return os.path.isfile(p) and os.path.getsize(p) > 0
def _read_csv_rows(p):
    if not _csv_exists(p): return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
def _append_row_csv(path, row, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    newfile = not _csv_exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if newfile: w.writeheader()
        w.writerow(row)
def _write_full_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
def _unique_key(lam, wr): return (round(float(lam),12), round(float(wr),12))
def _scale_range(t, factor): return (t[0]*factor, t[1]*factor)
def _parse_int_list(s): 
    if not s: return tuple()
    return tuple(int(x) for x in s.split(",") if x.strip())
def _parse_float_list(s):
    if not s: return []
    return [float(x) for x in s.split(",") if x.strip()]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# ----------------- Task length & epochs policy -----------------
TASK_MAXLEN = {"sst2":128, "cola":128, "mrpc":256, "rte":256, "stsb":256, "qnli":256, "mnli":512, "qqp":512}
SUPPORTED_TASKS = {"sst2","mrpc","cola","qnli","mnli","qqp","rte"}
FINAL_EPOCHS_PAPER = {"mnli":1,"qnli":1,"sst2":10,"mrpc":10,"cola":10,"qqp":10,"rte":10}

# ===== Task-aware priors (base at r=8) =====
LAMBDA_PRIORS_BASE = {
    "mnli": (5e-6, 8e-5),
    "qnli": (5e-6, 8e-5),
    "qqp" : (5e-5, 3e-4),
    # tighten for sst2/cola as discussed
    "sst2": (2e-5, 8e-5),
    "cola": (2e-5, 8e-5),
    "mrpc": (5e-5, 3e-4),
    "rte" : (5e-5, 3e-4),
}
WR_PRIORS = {"mnli": (0.25, 0.50), "qnli": (0.25, 0.50)}

# Default paired grids when user doesn't provide custom points (deterministic, no random)
HANDPICKED = {
    "mnli": [(7e-6,0.30),(1e-5,0.30),(2e-5,0.30),(3e-5,0.30)],
    "qnli": [(7e-6,0.30),(1e-5,0.30),(2e-5,0.30),(3e-5,0.30)],
    "sst2": [(2e-5,0.30),(3e-5,0.30),(5e-5,0.30)],
    "cola": [(2e-5,0.30),(3e-5,0.30),(5e-5,0.30)],
    "mrpc": [(2e-5,0.30),(3e-5,0.30),(5e-5,0.30)],
    "rte" : [(2e-5,0.30),(3e-5,0.30),(5e-5,0.30)],
}

def _rank_scaled_prior(task, lora_r, rank_scale_lambda):
    base = LAMBDA_PRIORS_BASE.get(task, (1e-4,5e-4))
    if rank_scale_lambda and lora_r != 8:
        return _scale_range(base, 8.0/max(lora_r,1))
    return base

def STAGEA_EPOCHS(task): return 1 if FINAL_EPOCHS_PAPER.get(task,10)==1 else 2
def STAGEB_EPOCHS(task): return FINAL_EPOCHS_PAPER.get(task,10)
def STAGEC_EPOCHS(task): return FINAL_EPOCHS_PAPER.get(task,10)

# ----------------- W&B env -----------------
# W&B logging is disabled by default for anonymous review artifacts.
# Enable external logging only with a non-identifying project name.
os.environ.setdefault("WANDB_WATCH", "false")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
def _set_wandb_project_for_task(task, version_suffix=""):
    if os.environ.get("USE_WANDB", "0") == "1":
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_PROJECT"] = f"pacf_glue_{task.lower()}{f'_{version_suffix}' if version_suffix else ''}"

# ----------------- CFG -----------------
@dataclass
class CFG:
    task_name: str = "sst2"
    model_name: str = "t5-base"
    max_length: int = 128
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05  # <- default 0.05
    # training
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.0   # <- default 0.0
    batch_size: int = 16
    eval_batch_size: int = 64
    warmup_ratio: float = 0.30
    logging_steps: int = 50
    train_frac: float = 1.0
    seed: int = 2
    # tweaks
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    # Flat-LoRA
    flat_sigma_max: float = 0.05
    # PACF
    pac_lambda: float = 3e-5
    pac_prior_var: float = 1.0
    pac_post_var: float = 1.0
    reg_warmup_ratio: float = 0.30
    mnli_split: str = "matched"
    @property
    def pac_warmup_ratio(self): return self.reg_warmup_ratio
    @pac_warmup_ratio.setter
    def pac_warmup_ratio(self, v): self.reg_warmup_ratio = v

CFG = CFG()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# -------- globals --------
tokenizer = None
data_collator = None
train_ds = None
eval_ds = None

# ----------------- Data -----------------
def switch_to_t5_glue():
    global tokenizer, data_collator, train_ds, eval_ds
    task = CFG.task_name.lower()
    if task not in TASK_MAXLEN: raise ValueError(f"Unsupported task '{task}'.")
    if task == "stsb": raise NotImplementedError("STS-B regression not supported.")
    CFG.max_length = TASK_MAXLEN[task]
    CFG.model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

    def preprocess(ex):
        t = task
        if t == "sst2":
            src = [f"sst2 sentence: {s}" for s in ex["sentence"]]
            tgt = ["negative" if y==0 else "positive" for y in ex["label"]]
        elif t == "mrpc":
            src = [f"mrpc s1: {a} s2: {b}" for a,b in zip(ex["sentence1"], ex["sentence2"])]
            tgt = ["not_equivalent" if y==0 else "equivalent" for y in ex["label"]]
        elif t == "cola":
            src = [f"cola sentence: {s}" for s in ex["sentence"]]
            tgt = ["unacceptable" if y==0 else "acceptable" for y in ex["label"]]
        elif t == "qnli":
            src = [f"qnli question: {q} sentence: {s}" for q, s in zip(ex["question"], ex["sentence"])]
            tgt = ["entailment" if y==0 else "not_entailment" for y in ex["label"]]
        elif t == "mnli":
            src = [f"mnli premise: {p} hypothesis: {h}" for p, h in zip(ex["premise"], ex["hypothesis"])]
            tgt = ["entailment" if y==0 else ("neutral" if y==1 else "contradiction") for y in ex["label"]]
        elif t == "qqp":
            src = [f"qqp q1: {a} q2: {b}" for a,b in zip(ex["question1"], ex["question2"])]
            tgt = ["not_duplicate" if y==0 else "duplicate" for y in ex["label"]]
        elif t == "rte":
            src = [f"rte premise: {p} hypothesis: {h}" for p,h in zip(ex["sentence1"], ex["sentence2"])]
            tgt = ["entailment" if y==0 else "not_entailment" for y in ex["label"]]
        else:
            raise ValueError(f"Unsupported task: {t}")
        model_inputs = tokenizer(src, truncation=True, max_length=CFG.max_length)
        labels = tokenizer(text_target=tgt, truncation=True, max_length=8)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw = load_dataset("glue", task)
    tr = raw["train"].map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    if task == "mnli":
        val_key = "validation_matched" if getattr(CFG, "mnli_split", "matched") == "matched" else "validation_mismatched"
    else:
        val_key = "validation" if "validation" in raw else "validation_matched"
    ev = raw[val_key].map(preprocess, batched=True, remove_columns=raw[val_key].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100)
    train_ds, eval_ds = tr, ev
    print(f"[OK] T5 pipeline ready: task={CFG.task_name} | max_len={CFG.max_length} | train={len(train_ds)} | eval={len(eval_ds)}")

# ----------------- Model -----------------
def build_lora_model():
    base = T5ForConditionalGeneration.from_pretrained(CFG.model_name)
    peft_cfg = LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout,
        target_modules=["q","k","v","o","wi_0","wi_1","wo"],
        bias="none", task_type="SEQ_2_SEQ_LM",
    )
    return get_peft_model(base, peft_cfg)

# ----------------- Metrics -----------------
metric_acc = _load_metric("accuracy")
metric_f1  = _load_metric("f1")
metric_mcc = _load_metric("matthews_correlation")
def _main_key_for_task(task):
    t = task.lower()
    if t == "cola": return "eval_mcc"
    if t == "mrpc": return "eval_f1"
    return "eval_accuracy"

def compute_metrics_t5(eval_pred):
    preds, labels = eval_pred
    if isinstance(labels, np.ndarray):
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    pred_str  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    t = CFG.task_name.lower()
    norm = lambda s: s.strip().lower().replace(" ","_")
    maps = {
        "sst2":{"negative":0,"positive":1},
        "mrpc":{"not_equivalent":0,"equivalent":1},
        "cola":{"unacceptable":0,"acceptable":1},
        "qnli":{"entailment":0,"not_entailment":1},
        "mnli":{"entailment":0,"neutral":1,"contradiction":2},
        "qqp":{"not_duplicate":0,"duplicate":1},
        "rte":{"entailment":0,"not_entailment":1},
    }
    str2id = maps[t]; default_label = list(str2id.values())[0]
    y_pred = [str2id.get(norm(s), default_label) for s in pred_str]
    y_true = [str2id.get(norm(s), default_label) for s in label_str]
    out = {"accuracy": metric_acc.compute(predictions=y_pred, references=y_true)["accuracy"]}
    if t in {"sst2","mrpc","qqp","rte","qnli","cola"}:
        out["f1"]  = metric_f1.compute(predictions=y_pred, references=y_true, average="binary")["f1"]
        out["mcc"] = metric_mcc.compute(predictions=y_pred, references=y_true)["matthews_correlation"]
    if t == "mnli":
        out["f1"] = metric_f1.compute(predictions=y_pred, references=y_true, average="macro")["f1"]
    return out

# ----------------- PAC KL -----------------
def pac_kl_and_loss(model, prior_var=1.0, post_var=1.0):
    theta_sq_sum, count = 0.0, 0
    for n, p in model.named_parameters():
        if "lora_" in n and p.requires_grad:
            theta_sq_sum += (p.float()**2).sum()
            count += p.numel()
    if count == 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device), 0.0, 0.0
    device = next(model.parameters()).device
    kl = 0.5 * (theta_sq_sum/prior_var + (count*post_var)/prior_var - count - count*math.log(post_var/prior_var))
    kl_per_dim = kl.item() / count
    kl_eff = 0.5 * (theta_sq_sum/prior_var)
    kl_eff_per_dim = kl_eff.item() / count
    return kl, kl_per_dim, kl_eff_per_dim

# ----------------- Flat-LoRA callback -----------------
class FlatLoRAInjectCallback_RepoStyle(TrainerCallback):
    def __init__(self, total_opt_steps, rho=0.05, log_every=200, base_seed=3):
        self.total_opt_steps = max(1, int(total_opt_steps))
        self.rho = float(rho); self.log_every = int(log_every); self.base_seed = int(base_seed)
        self._armed_step = None; self._cached_filter_norms = None
    def _cosine_factor(self, current_opt_step):
        x = min(max(current_opt_step / self.total_opt_steps, 0.0), 1.0)
        return 0.5*(1.0-math.cos(math.pi*x))
    @torch.no_grad()
    def _iter_lora_linear(self, model):
        for m in model.modules():
            if isinstance(m, LoraLinear): yield m
    @torch.no_grad()
    def _compute_filter_norms(self, model, factor):
        norms = []
        for module in self._iter_lora_linear(model):
            W = module.weight.data
            A = module.lora_A['default'].weight.data
            B = module.lora_B['default'].weight.data
            scaling = module.scaling['default']
            Wp = W + scaling * (B @ A)
            n_in = max(1, Wp.shape[1])
            row_norm = torch.norm(Wp, p=2, dim=1, keepdim=True)
            std = (factor * self.rho) / math.sqrt(n_in) * row_norm
            norms.append(std.to(Wp.device, dtype=Wp.dtype))
        return norms
    @staticmethod
    def _step_seed(base_seed, step): return (hash((base_seed, int(step))) & 0xFFFFFFFF)
    @torch.no_grad()
    def _apply_noise(self, model, step, filter_norms, sign):
        torch.manual_seed(self._step_seed(self.base_seed, step))
        idx = 0
        for module in self._iter_lora_linear(model):
            md = module.weight.data
            std = filter_norms[idx].expand_as(md).to(device=md.device, dtype=md.dtype)
            noise = torch.randn_like(md) * std
            md.add_(sign * noise); idx += 1
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]; current_opt_step = state.global_step
        if self._armed_step is None or self._armed_step < current_opt_step:
            factor = self._cosine_factor(current_opt_step)
            self._cached_filter_norms = self._compute_filter_norms(model, factor)
            self._apply_noise(model, current_opt_step, self._cached_filter_norms, sign=+1)
            self._armed_step = current_opt_step
    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]; current_opt_step = state.global_step
        if self._armed_step == current_opt_step and self._cached_filter_norms:
            self._apply_noise(model, current_opt_step, self._cached_filter_norms, sign=-1)
            self._cached_filter_norms = None
    def on_step_end(self, args, state, control, **kwargs):
        if self._cached_filter_norms is not None:
            self._cached_filter_norms = None

# ----------------- PAC Trainer -----------------
class PACSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, pac_lambda=1e-3, pac_prior_var=1.0, pac_post_var=1.0,
                 reg_warmup_ratio=0.2, total_steps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pac_lambda = float(pac_lambda)
        self.pac_prior_var = float(pac_prior_var)
        self.pac_post_var = float(pac_post_var)
        self.reg_warmup_ratio = float(reg_warmup_ratio)
        if total_steps is None:
            steps_per_epoch = max(1, math.ceil(len(self.train_dataset) / self.args.per_device_train_batch_size))
            total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        self._total_steps = max(1, int(total_steps))
    def _pac_weight(self):
        step = getattr(self.state, "global_step", 0)
        warm = max(1, int(self.reg_warmup_ratio * self._total_steps))
        x = min(max(step / warm, 0.0), 1.0)
        return 0.5 * (1.0 - math.cos(math.pi * x))
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce = outputs.loss
        kl, kl_per_dim, kl_eff_per_dim = pac_kl_and_loss(model, self.pac_prior_var, self.pac_post_var)
        loss = ce + (self.pac_lambda * self._pac_weight()) * kl if model.training else ce
        try:
            self.log({
                "kl_eff_per_dim": float(kl_eff_per_dim),
                "kl_full_per_dim": float(kl_per_dim),
                "pac_lambda": float(self.pac_lambda),
                "reg_warmup_ratio": float(self.reg_warmup_ratio),
                "pac_warmup_ratio": float(self.reg_warmup_ratio),
                "pac_weight": float(self._pac_weight()) if model.training else 0.0
            })
        except Exception:
            pass
        return loss

# ----------------- Trainer args -----------------
def _make_args(run_name=None):
    return Seq2SeqTrainingArguments(
        output_dir=f"runs/{CFG.task_name}",
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.eval_batch_size,
        learning_rate=CFG.lr,
        num_train_epochs=CFG.epochs,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        logging_steps=CFG.logging_steps,
        save_strategy="no",
        predict_with_generate=True,
        report_to=[],
        run_name=run_name,
        seed=CFG.seed,
        data_seed=CFG.seed,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        max_grad_norm=CFG.max_grad_norm,
        label_smoothing_factor=CFG.label_smoothing,
        lr_scheduler_type=CFG.lr_scheduler_type,
        bf16=False, fp16=False,
        generation_max_length=8,
        eval_strategy="epoch",
    )

def _total_steps_estimate(train_len, per_device_bz, epochs, grad_acc=1):
    steps_per_epoch = math.ceil(train_len / max(1, per_device_bz))
    opt_steps_per_epoch = math.ceil(steps_per_epoch / max(1, grad_acc))
    return int(opt_steps_per_epoch * epochs)

# ----------------- Run one variant -----------------
def run_variant(variant):
    assert variant in {"lora","flat_lora","lora_pacf"}
    set_seed(CFG.seed)
    os.environ.setdefault("WANDB_RUN_GROUP", f"{CFG.task_name}-r{CFG.lora_r}-e{CFG.epochs}")
    run_name = (f"{CFG.task_name}-r{CFG.lora_r}-{variant}"
                f"-seed{CFG.seed}-e{CFG.epochs}-lr{CFG.lr}"
                f"-pac{CFG.pac_lambda}-pv{CFG.pac_prior_var}-wr{CFG.reg_warmup_ratio}"
                f"-drop{CFG.lora_dropout}-wd{CFG.weight_decay}"
                f"-mnli_{CFG.mnli_split if CFG.task_name.lower()=='mnli' else 'na'}")
    model = build_lora_model()
    args = _make_args(run_name=run_name)

    callbacks = []
    if variant == "flat_lora":
        tot_opt = _total_steps_estimate(len(train_ds), args.per_device_train_batch_size, CFG.epochs, args.gradient_accumulation_steps)
        callbacks.append(FlatLoRAInjectCallback_RepoStyle(total_opt_steps=tot_opt, rho=CFG.flat_sigma_max, log_every=200, base_seed=CFG.seed))

    TrainerCls = PACSeq2SeqTrainer if variant == "lora_pacf" else Seq2SeqTrainer
    extra_kwargs = {}
    if variant == "lora_pacf":
        tot = _total_steps_estimate(len(train_ds), args.per_device_train_batch_size, CFG.epochs, args.gradient_accumulation_steps)
        extra_kwargs.update(dict(pac_lambda=CFG.pac_lambda, pac_prior_var=CFG.pac_prior_var,
                                 pac_post_var=CFG.pac_post_var, reg_warmup_ratio=CFG.reg_warmup_ratio, total_steps=tot))
    try:
        trainer = TrainerCls(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tokenizer, data_collator=data_collator,
                             compute_metrics=compute_metrics_t5, callbacks=callbacks, **extra_kwargs)
    except TypeError:
        trainer = TrainerCls(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                             tokenizer=tokenizer, data_collator=data_collator,
                             compute_metrics=compute_metrics_t5, callbacks=callbacks, **extra_kwargs)
    setattr(trainer, "label_names", ["labels"])
    trainer.train()
    ev = trainer.evaluate()

    # log to W&B (best effort)
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "task": CFG.task_name, "lr": CFG.lr, "epochs": CFG.epochs, "seed": CFG.seed,
                "scheduler": CFG.lr_scheduler_type, "max_length": CFG.max_length,
                "warmup_ratio": CFG.warmup_ratio, "weight_decay": CFG.weight_decay,
                "lora_r": CFG.lora_r, "lora_alpha": CFG.lora_alpha, "lora_dropout": CFG.lora_dropout,
                "flat_sigma_max": CFG.flat_sigma_max, "pac_lambda": CFG.pac_lambda,
                "pac_prior_var": CFG.pac_prior_var, "pac_post_var": CFG.pac_post_var,
                "reg_warmup_ratio": CFG.reg_warmup_ratio,
                "mnli_split": CFG.mnli_split if CFG.task_name.lower()=="mnli" else "na",
            }, allow_val_change=True)
            wandb.finish()
    except Exception:
        pass

    # collect extra
    extra_out = {}
    if variant == "lora_pacf":
        hist = getattr(trainer.state, "log_history", []) or []
        kl_eff = kl_full = None
        for rec in reversed(hist):
            if "kl_eff_per_dim" in rec and kl_eff is None:  kl_eff  = rec["kl_eff_per_dim"]
            if "kl_full_per_dim" in rec and kl_full is None: kl_full = rec["kl_full_per_dim"]
            if kl_eff is not None and kl_full is not None: break
        extra_out["kl_eff_per_dim"]  = float(kl_eff)  if kl_eff  is not None else None
        extra_out["kl_full_per_dim"] = float(kl_full) if kl_full is not None else None

    # eval_cache
    try:
        os.makedirs(os.path.join("runs","eval_cache"), exist_ok=True)
        cache_path = os.path.join("runs","eval_cache","eval_cache.csv")
        row = {
            "timestamp": _now_tag(),
            "run_name": run_name,
            "task": CFG.task_name,
            "rank": CFG.lora_r,
            "variant": variant,
            "seed": CFG.seed,
            "epochs": CFG.epochs,
            "eval_accuracy": ev.get("eval_accuracy"),
            "eval_f1": ev.get("eval_f1"),
            "eval_mcc": ev.get("eval_mcc"),
            "eval_loss": ev.get("eval_loss"),
            "kl_eff_per_dim": extra_out.get("kl_eff_per_dim") if variant=="lora_pacf" else None,
            "kl_full_per_dim": extra_out.get("kl_full_per_dim") if variant=="lora_pacf" else None,
            "flat_sigma_max": CFG.flat_sigma_max,
            "pac_lambda": CFG.pac_lambda if variant=="lora_pacf" else None,
            "reg_warmup_ratio": CFG.reg_warmup_ratio if variant=="lora_pacf" else None,
            "lora_alpha": CFG.lora_alpha,
            "lora_r": CFG.lora_r,
            "model_name": CFG.model_name,
            "lora_dropout": CFG.lora_dropout,
            "weight_decay": CFG.weight_decay,
            "lr": CFG.lr,
            "scheduler": CFG.lr_scheduler_type,
            "mnli_split": CFG.mnli_split if CFG.task_name.lower()=="mnli" else "na",
        }
        # eval_cache
        write_header = not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0
        with open(cache_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print("[WARN] failed to write eval_cache:", e)

    return ev, extra_out

# ----------------- Paper-style compare (from cache) -----------------
def paper_style_compare_from_cache(task="mnli", r=8, seeds=(1,2,3), save=True, note=""):
    import numpy as _np
    task = task.lower()
    cache_path = os.path.join("runs", "eval_cache", "eval_cache.csv")
    if not os.path.exists(cache_path):
        print(f"[ERR] cache not found: {cache_path}"); return {}
    rows = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            try:
                if rec.get("task","").lower()!=task: continue
                if int(float(rec.get("rank",-1)))!=int(r): continue
                if rec.get("variant") not in {"lora","flat_lora","lora_pacf"}: continue
                if int(float(rec.get("seed",-1))) not in set(map(int, seeds)): continue
                def _to_float(x): return None if x in (None,"","None") else float(x)
                for k in ["eval_accuracy","eval_f1","eval_loss","kl_eff_per_dim","kl_full_per_dim","flat_sigma_max","pac_lambda","reg_warmup_ratio","lora_alpha","lora_r","weight_decay","lr"]:
                    rec[k] = _to_float(rec.get(k))
                rec["epochs"] = int(float(rec.get("epochs",0) or 0))
                rows.append(rec)
            except: continue
    if not rows:
        print("[WARN] no cache rows matched your filters."); return {}

    latest = {}
    for rec in rows:
        key = (rec["variant"], int(float(rec["seed"])))
        if key not in latest or rec.get("timestamp","") > latest[key].get("timestamp",""):
            latest[key] = rec
    rows = list(latest.values())

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs: return float("nan"), 0.0
        m = float(_np.mean(xs)); s = float(_np.std(xs, ddof=1)) if len(xs)>1 else 0.0
        return m, s

    summary, details = {}, []
    for variant in ["lora","flat_lora","lora_pacf"]:
        sub = [r1 for r1 in rows if r1["variant"]==variant]
        acc_m, acc_s = ms([r1.get("eval_accuracy") for r1 in sub])
        f1_m,  f1_s  = ms([r1.get("eval_f1")       for r1 in sub])
        loss_m,loss_s= ms([r1.get("eval_loss")     for r1 in sub])
        summary[variant] = {"acc_mean":acc_m,"acc_std":acc_s,"f1_mean":f1_m,"f1_std":f1_s,"loss_mean":loss_m,"loss_std":loss_s}
        details.extend(sub)

    if save:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join("runs", "paper_compare", f"{task}_t5_r{r}_e{CFG.epochs}_{stamp}")
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "details.csv"), "w", newline="", encoding="utf-8") as f:
            fields = ["task","rank","variant","seed","eval_accuracy","eval_f1","eval_loss","kl_eff_per_dim","kl_full_per_dim","epochs","flat_sigma_max","pac_lambda","reg_warmup_ratio","lora_alpha","lora_r","model_name","lora_dropout","weight_decay","lr","scheduler","mnli_split","run_name","timestamp"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for rec in details: w.writerow({k: rec.get(k) for k in fields})
        with open(os.path.join(base, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["variant","acc_mean","acc_std","f1_mean","f1_std","loss_mean","loss_std"])
            for v in ["lora","flat_lora","lora_pacf"]:
                o = summary[v]; w.writerow([v,o["acc_mean"],o["acc_std"],o["f1_mean"],o["f1_std"],o["loss_mean"],o["loss_std"]])
        meta = {"task":task, "rank":r, "stamp":stamp, "cfg":asdict(CFG),
                "versions":{"python":platform.python_version(),"torch":torch.__version__,"transformers":__import__("transformers").__version__},
                "note": note or "aggregated from eval_cache (no retrain)"}
        with open(os.path.join(base, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
        print(f"[OK] saved to: {base}")
    return summary

# ----------------- Auto pipeline (Stage A/B/C) -----------------
def auto_run(task, ranks=[16], topk_to_stageB=3, stop_after_stage="C",
             grid_lambda=None, grid_wr=None, grid_lambda_points=None, grid_wr_points=None,
             rank_scale_lambda=True, batch_size=16, version_suffix="", seeds_stageC=(1,2,3),
             skip_stageA=False):
    task = task.lower()
    CFG.task_name = task
    CFG.batch_size = batch_size
    CFG.eval_batch_size = max(64, batch_size)
    _set_wandb_project_for_task(task, version_suffix=version_suffix)
    switch_to_t5_glue()
    main_key = _main_key_for_task(task)

    stageA_all = []
    for r in ranks:
        CFG.lora_r = r
        _ = _rank_scaled_prior(task, r, rank_scale_lambda)

        # Stage-A seeds/epochs
        if skip_stageA:
            if grid_lambda_points and grid_wr_points:
                candidate_pairs = [(float(l), float(w)) for l in grid_lambda_points for w in grid_wr_points]
            else:
                candidate_pairs = HANDPICKED.get(task, [])
            if not candidate_pairs:
                raise ValueError("Value_error")
        
            topk_rows = [{"pac_lambda": l, "reg_warmup_ratio": w, "metric": 0.0} for (l, w) in candidate_pairs][:topk_to_stageB]
            stageA_all.append({"rank": r, "topk": topk_rows})
            continue
        CFG.epochs = STAGEA_EPOCHS(task)
        CFG.seed = 2
        os.environ["WANDB_RUN_GROUP"] = f"{task}-StageA-{CFG.epochs}e-r{r}"
        base = os.path.join("runs", "stageA", f"{task}_r{r}")
        os.makedirs(base, exist_ok=True)

        # Flat baseline（seed=2）
        flat_csv = os.path.join(base, "stageA_flat_baseline.csv")
        if not _csv_exists(flat_csv):
            ev_flat, _ = run_variant("flat_lora")
            _append_row_csv(flat_csv, {
                "timestamp": _now_tag(), "rank": r, "seed": CFG.seed, "epochs": CFG.epochs,
                "variant": "flat_lora", "eval_accuracy": ev_flat.get("eval_accuracy"),
                "eval_f1": ev_flat.get("eval_f1"), "eval_mcc": ev_flat.get("eval_mcc"),
                "eval_loss": ev_flat.get("eval_loss")
            }, ["timestamp","rank","seed","epochs","variant","eval_accuracy","eval_f1","eval_mcc","eval_loss"])

        trials_csv = os.path.join(base, "stageA_trials.csv")
        trials_fields = ["timestamp","trial","rank","seed","epochs","pac_lambda","reg_warmup_ratio",
                         "metric","eval_accuracy","eval_f1","eval_mcc","eval_loss","lambda_report","wr_report"]

        candidate_pairs = []
        if grid_lambda_points and grid_wr_points:
            lams = grid_lambda_points
            wrs  = grid_wr_points
            for lam in lams:
                for wr in wrs:
                    candidate_pairs.append((float(lam), float(wr)))
        else:
            candidate_pairs = HANDPICKED.get(task, [])
            if not candidate_pairs:
                low, high = _rank_scaled_prior(task, r, rank_scale_lambda)
                lams = np.geomspace(low, high, 3)
                candidate_pairs = [(float(l), 0.30) for l in lams]

        seen = set((round(float(p[0]),12), round(float(p[1]),12)) for p in candidate_pairs)
        prev_rows = _read_csv_rows(trials_csv)
        done = set()
        for r0 in prev_rows:
            try:
                done.add(_unique_key(float(r0["pac_lambda"]), float(r0["reg_warmup_ratio"])))
            except: pass

        trial_idx = len(prev_rows)
        all_rows = prev_rows[:]
        for lam, wr in candidate_pairs:
            key = _unique_key(lam, wr)
            if key in done: continue
            CFG.pac_lambda = float(lam); CFG.reg_warmup_ratio = float(wr)
            ev, _ = run_variant("lora_pacf")
            trial_idx += 1
            row = {
                "timestamp": _now_tag(), "trial": trial_idx, "rank": r, "seed": CFG.seed, "epochs": CFG.epochs,
                "pac_lambda": CFG.pac_lambda, "reg_warmup_ratio": CFG.reg_warmup_ratio,
                "metric": ev.get(main_key, 0.0), "eval_accuracy": ev.get("eval_accuracy"),
                "eval_f1": ev.get("eval_f1"), "eval_mcc": ev.get("eval_mcc"), "eval_loss": ev.get("eval_loss"),
                "lambda_report": f"{CFG.pac_lambda:.2e}", "wr_report": f"{CFG.reg_warmup_ratio:.2f}",
            }
            _append_row_csv(trials_csv, row, trials_fields)
            all_rows.append(row)
            _write_full_csv(os.path.join(base, f"stageA_trials_{_now_tag()}.csv"), all_rows, trials_fields)

        all_rows = _read_csv_rows(trials_csv)
        if not all_rows:
            raise RuntimeError("Stage-A produced no candidates.")
        topk_rows = sorted(all_rows, key=lambda x: float(x.get("metric",0.0)), reverse=True)[:topk_to_stageB]
        stageA_all.append({"rank": r, "topk": topk_rows})

        if stop_after_stage.upper()=="A": print("[Auto] stop_after_stage=A"); return

    # ---- Stage B ----
    stageB_all = []
    for pack in stageA_all:
        r = pack["rank"]
        CFG.lora_r = r; CFG.epochs = STAGEB_EPOCHS(task); CFG.seed = 2
        os.environ["WANDB_RUN_GROUP"] = f"{task}-StageB-{CFG.epochs}e-r{r}"
        base = os.path.join("runs", "stageB", f"{task}_r{r}")
        os.makedirs(base, exist_ok=True)

        # Flat baseline（seed=2）
        flat_csv = os.path.join(base, "stageB_flat_baseline.csv")
        if not _csv_exists(flat_csv):
            ev_flat, _ = run_variant("flat_lora")
            _append_row_csv(flat_csv, {
                "timestamp": _now_tag(), "rank": r, "seed": CFG.seed, "epochs": CFG.epochs,
                "variant": "flat_lora", "eval_accuracy": ev_flat.get("eval_accuracy"),
                "eval_f1": ev_flat.get("eval_f1"), "eval_mcc": ev_flat.get("eval_mcc"),
                "eval_loss": ev_flat.get("eval_loss")
            }, ["timestamp","rank","seed","epochs","variant","eval_accuracy","eval_f1","eval_mcc","eval_loss"])

        cand_csv = os.path.join(base, "stageB_candidates.csv")
        cand_fields = ["timestamp","candidate","rank","seed","epochs","pac_lambda","reg_warmup_ratio","metric",
                       "eval_accuracy","eval_f1","eval_mcc","eval_loss","lambda_report","wr_report"]
        prev = _read_csv_rows(cand_csv)
        done = set()
        for r0 in prev:
            try: done.add(_unique_key(float(r0["pac_lambda"]), float(r0["reg_warmup_ratio"])))
            except: pass
        rows_all = prev[:]

        uniq, seen_local = [], set()
        for x in pack["topk"]:
            lam, wr = float(x["pac_lambda"]), float(x["reg_warmup_ratio"])
            key = _unique_key(lam, wr)
            if key not in seen_local:
                seen_local.add(key); uniq.append((lam, wr))

        cand_idx = len(prev)
        for lam, wr in uniq:
            key = _unique_key(lam, wr)
            if key in done: continue
            CFG.pac_lambda = float(lam); CFG.reg_warmup_ratio = float(wr)
            ev, _ = run_variant("lora_pacf")
            cand_idx += 1
            row = {"timestamp": _now_tag(), "candidate": cand_idx, "rank": r, "seed": CFG.seed, "epochs": CFG.epochs,
                   "pac_lambda": CFG.pac_lambda, "reg_warmup_ratio": CFG.reg_warmup_ratio, "metric": ev.get(main_key, 0.0),
                   "eval_accuracy": ev.get("eval_accuracy"), "eval_f1": ev.get("eval_f1"),
                   "eval_mcc": ev.get("eval_mcc"), "eval_loss": ev.get("eval_loss"),
                   "lambda_report": f"{CFG.pac_lambda:.2e}", "wr_report": f"{CFG.reg_warmup_ratio:.2f}"}
            _append_row_csv(cand_csv, row, cand_fields)
            rows_all.append(row)
            _write_full_csv(os.path.join(base, f"stageB_candidates_{_now_tag()}.csv"), rows_all, cand_fields)

        best = sorted(_read_csv_rows(cand_csv), key=lambda x: (float(x.get("metric",0.0)),-float(x.get("eval_loss",0.0)) if x.get("eval_loss") else 0.0), reverse=True)[0]
        stageB_all.append({"rank": r, "best": best})
        if stop_after_stage.upper()=="B": print("[Auto] stop_after_stage=B"); return

    # ---- Stage C ----
    print("Stage_C")
    for pack in stageB_all:
        r = pack["rank"]; CFG.lora_r = r; CFG.epochs = STAGEC_EPOCHS(task)
        best_lambda = float(pack["best"]["pac_lambda"]); best_wr = float(pack["best"]["reg_warmup_ratio"])

        # PACF(best)
        os.environ["WANDB_RUN_GROUP"] = f"{task}-StageC-{CFG.epochs}e-r{r}-PACF"
        for s in seeds_stageC:
            CFG.seed = int(s); CFG.pac_lambda = best_lambda; CFG.reg_warmup_ratio = best_wr
            run_variant("lora_pacf")

        # Baselines
        for variant in ["lora","flat_lora"]:
            os.environ["WANDB_RUN_GROUP"] = f"{task}-StageC-{CFG.epochs}e-r{r}-{variant}"
            for s in seeds_stageC:
                CFG.seed = int(s); run_variant(variant)

        paper_style_compare_from_cache(task=task, r=r, seeds=seeds_stageC, save=True,
            note=f"A=grid({len(HANDPICKED.get(task,[])) or 'custom'}) , B={len(stageB_all)}, seedAB=2; best(lam,wr)=({best_lambda:.2e},{best_wr:.2f})")

def run_three_variants_then_summary(task, r, epochs, seeds, pac_lambda, wr):
    CFG.task_name = task; CFG.lora_r = int(r); CFG.eval_batch_size = max(64, CFG.batch_size)
    switch_to_t5_glue()
    for variant in ["lora","flat_lora"]:
        for s in seeds:
            CFG.seed = int(s); CFG.epochs = int(epochs) if epochs is not None else FINAL_EPOCHS_PAPER.get(task,10)
            run_variant(variant)
    CFG.pac_lambda = float(pac_lambda); CFG.reg_warmup_ratio = float(wr)
    for s in seeds:
        CFG.seed = int(s); CFG.epochs = int(epochs) if epochs is not None else FINAL_EPOCHS_PAPER.get(task,10)
        run_variant("lora_pacf")
    _ = paper_style_compare_from_cache(task=task, r=int(r), seeds=tuple(map(int, seeds)), save=True,
                                       note=f"paper one-click; λ={pac_lambda:.2e}, wr={wr:.2f}")
    print("[OK] paper-style summary saved.")

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="auto", choices=["auto","single","seeds","compare-cache","paper"])
    ap.add_argument("--task", type=str, default="sst2", choices=list(SUPPORTED_TASKS))
    ap.add_argument("--ranks", type=str, default="16")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)

    # variant (single / seeds)
    ap.add_argument("--variant", type=str, default="lora_pacf", choices=["lora","flat_lora","lora_pacf"])

    # PACF (single/seeds)
    ap.add_argument("--pac_lambda", type=float, default=None)
    ap.add_argument("--wr", type=float, default=None)

    # Grid control (auto)
    ap.add_argument("--grid_lambda_points", type=str, default="", help="e.g. '2e-5,3e-5,5e-5'")
    ap.add_argument("--grid_wr_points", type=str, default="", help="e.g. '0.28,0.30,0.34'")
    ap.add_argument("--no_rank_scale_lambda", action="store_true")

    # optimizer & lora
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--flat_sigma_max", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.30)
    ap.add_argument("--scheduler", type=str, default="cosine")

    # auto A/B/C
    ap.add_argument("--topk_to_stageB", type=int, default=3)
    ap.add_argument("--stop_after_stage", type=str, default="C", choices=["A","B","C"])
    ap.add_argument("--version_suffix", type=str, default="")
    ap.add_argument("--skip_stageA", action="store_true")
    ap.add_argument("--seeds_stageC", type=str, default="1,2,3")
    ap.add_argument("--auto_summary", action="store_true")

    args = ap.parse_args()

    # set CFG from args
    CFG.task_name = args.task
    CFG.lora_r = args.lora_r
    CFG.lora_alpha = args.lora_alpha
    CFG.lora_dropout = args.lora_dropout
    CFG.lr = args.lr
    CFG.weight_decay = args.wd
    CFG.batch_size = args.batch_size
    CFG.eval_batch_size = max(64, args.batch_size)
    CFG.warmup_ratio = args.warmup_ratio
    CFG.lr_scheduler_type = args.scheduler
    CFG.flat_sigma_max = args.flat_sigma_max
    if args.epochs is not None: CFG.epochs = args.epochs

    if args.mode in {"single","seeds"}:
        _set_wandb_project_for_task(args.task, version_suffix=args.version_suffix)
        switch_to_t5_glue()
        if args.pac_lambda is not None: CFG.pac_lambda = args.pac_lambda
        if args.wr is not None: CFG.reg_warmup_ratio = args.wr

    if args.mode == "single":
        CFG.seed = args.seed
        print(f"[Run] single variant={args.variant} task={args.task} r={CFG.lora_r} seed={CFG.seed}")
        run_variant(args.variant)

    elif args.mode == "seeds":
        CFG.epochs = CFG.epochs or FINAL_EPOCHS_PAPER.get(args.task, 10)
        print(f"[Run] seeds PACF task={args.task} r={CFG.lora_r} λ={CFG.pac_lambda} wr={CFG.reg_warmup_ratio}")
        for s in _parse_int_list(args.seeds):
            CFG.seed = s; run_variant("lora_pacf")
        if args.auto_summary:
            seeds_tuple = tuple(int(x) for x in args.seeds.split(",") if x.strip())
            _ = paper_style_compare_from_cache(task=args.task, r=CFG.lora_r, seeds=seeds_tuple, save=True,
                                               note="auto-summary from --mode seeds")
            print("[OK] auto summary saved (compare-cache).")

    elif args.mode == "compare-cache":
        seeds = _parse_int_list(args.seeds)
        _ = paper_style_compare_from_cache(task=args.task, r=args.lora_r, seeds=seeds, save=True, note="cli compare-cache")

    elif args.mode == "paper":
        seeds_tuple = _parse_int_list(args.seeds) or (1,2,3)
        if args.pac_lambda is None or args.wr is None:
            raise ValueError("Value_error_2")
        _set_wandb_project_for_task(args.task, version_suffix=args.version_suffix)
        run_three_variants_then_summary(task=args.task, r=args.lora_r, epochs=args.epochs, seeds=seeds_tuple,
                                        pac_lambda=float(args.pac_lambda), wr=float(args.wr))

    else:  # auto
        ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
        seeds_c = _parse_int_list(args.seeds_stageC) or (1,2,3)
        gl_points = _parse_float_list(args.grid_lambda_points)
        gw_points = _parse_float_list(args.grid_wr_points)
        _set_wandb_project_for_task(args.task, version_suffix=args.version_suffix)
        auto_run(task=args.task,
                 ranks=ranks,
                 topk_to_stageB=args.topk_to_stageB,
                 stop_after_stage=args.stop_after_stage,
                 grid_lambda=None, grid_wr=None,
                 grid_lambda_points=gl_points if gl_points else None,
                 grid_wr_points=gw_points if gw_points else None,
                 rank_scale_lambda=(not args.no_rank_scale_lambda),
                 batch_size=args.batch_size,
                 version_suffix=args.version_suffix,
                 seeds_stageC=seeds_c,
                 skip_stageA=args.skip_stageA)

if __name__ == "__main__":
    main()
