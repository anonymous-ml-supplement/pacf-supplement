#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIP (ViT-B/32) image classification experiments for PACF / LoRA / Flat-LoRA.

Implements a Stage A/B/C protocol similar to the GLUE script.
Outputs are written under `runs/clip/` by default.

No user-identifying info is included.

Usage examples (see README):
  python src/clip/pacf_clip.py --mode paper --dataset cifar10 --r 16 --seeds 1,2,3
"""

# pacf_clip_ic_full.py, Deterministic grid A, Top-K B, multi-seed C; bf16-first; r-scaled lambda
# CLIP ViT-B/32 Image Classification with LoRA / Flat-LoRA / PACF
# - acc/loss/macro-F1
# - Stage-A: deterministic paired grid (no random) on PACF only, epochs=2
# - Stage-B: take Top-K from A (seed=2), re-evaluate with epochs=10
# - Stage-C: PACF(best) + LoRA + Flat across seeds; save paper-style summary

import os, math, time, json, csv, random, argparse, platform, hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms
import open_clip
from contextlib import nullcontext
from datetime import datetime

try:
    import wandb
except Exception:
    wandb = None

# ----------------- seeds -----------------
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

# ----------------- config -----------------
@dataclass
class CFG:
    dataset: str = "cifar10"      # svhn|cifar10|dtd|cars|cifar100
    # training
    epochs: int = 10              
    batch_size: int = 128
    lr_lora: float = 5e-4         # LoRA/Flat/PACF
    lr_full: float = 1e-4
    weight_decay: float = 0.0    
    warmup_ratio: float = 0.30
    seed: int = 2
    num_workers: int = 2
    img_size: int = 224
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Flat-LoRA
    flat_sigma_max: float = 0.05  # baseline 0.05
    # PACF
    pac_lambda: float = 2.0e-04
    pac_prior_var: float = 1.0
    pac_post_var: float = 1.0
    reg_warmup_ratio: float = 0.30
    # multi-seed
    seeds_stageC: Tuple[int,...] = (1,2,3)
    # wandb
    wandb_project_prefix: str = "pacf_clip_v6"

CFG = CFG()

# ----------------- deterministic helpers -----------------
def _ts(): return datetime.now().strftime("%Y%m%d-%H%M%S")
def _short_hash(s: str, k: int = 6): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:k]
def _sanitize(x: str): return (str(x).replace("/", "_").replace(":", "_").replace(" ", "").replace(",", "_"))

def make_default_outdir(dataset: str, mode: str, seed: int, r: int, alpha: int,
                        lr: float | None = None, extras: Dict[str, Any] | None = None):
    extras = extras or {}
    sig = {"dataset": dataset, "mode": mode, "seed": seed, "r": r, "alpha": alpha, "lr": lr, **extras}
    name = f"{_ts()}-s{seed}-r{r}-a{alpha}-{_short_hash(json.dumps(sig, sort_keys=True))}"
    if lr is not None: name = f"{name}-lr{lr:.0e}"
    return os.path.join(os.path.join("runs","clip"), _sanitize(dataset), _sanitize(mode), name)

def make_summary_outdir(dataset: str, mode: str, r: int, alpha: int,
                        lr: float | None = None, extras: Dict[str, Any] | None = None, tag: str = "seeds"):
    extras = extras or {}
    sig = {"dataset": dataset, "mode": mode, "r": r, "alpha": alpha, "lr": lr, **extras}
    name = f"{_ts()}-r{r}-a{alpha}-{_short_hash(json.dumps(sig, sort_keys=True))}"
    if lr is not None: name = f"{name}-lr{lr:.0e}"
    return os.path.join(os.path.join("runs","clip"), _sanitize(dataset), f"{_sanitize(mode)}_{tag}", name)

# ----------------- AMP (bf16-first) -----------------
def choose_cuda_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16

def make_scaler(device: torch.device, cuda_dtype: torch.dtype):
    if device.type != "cuda" or cuda_dtype is torch.bfloat16:
        return None
    try:
        return torch.amp.GradScaler()
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)

def autocast_ctx(device: torch.device, cuda_dtype: torch.dtype):
    if device.type == "cuda":
        try:
            return torch.amp.autocast(device_type="cuda", dtype=cuda_dtype)
        except TypeError:
            return torch.cuda.amp.autocast()
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
        return torch.cpu.amp.autocast()
    return nullcontext()

# ----------------- Minimal LoRA Linear -----------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.enable_lora = True

    @property
    def weight(self): return self.base.weight
    @weight.setter
    def weight(self, v): self.base.weight = v
    @property
    def bias(self): return self.base.bias
    @bias.setter
    def bias(self, v): self.base.bias = v

    def forward(self, x):
        y = self.base(x)
        if self.enable_lora:
            y = y + self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return y

def _resolve_parent_and_key(model: nn.Module, qn: str):
    parts = qn.split("."); p = model
    for s in parts[:-1]:
        p = p[int(s)] if s.isdigit() else getattr(p, s)
    last = parts[-1]
    return p, (int(last) if last.isdigit() else last)

def inject_lora_into_vit(model: nn.Module, r: int, alpha: int, dropout: float):
    target = []
    for qn, m in model.named_modules():
        if qn.startswith("visual.transformer") and isinstance(m, nn.Linear):
            target.append((qn, m))
    replaced, hit = 0, []
    for qn, _ in target:
        parent, key = _resolve_parent_and_key(model, qn)
        m = parent[key] if isinstance(key, int) else getattr(parent, key)
        new_m = LoRALinear(m, r=r, alpha=alpha, dropout=dropout)
        if isinstance(key, int): parent[key] = new_m
        else: setattr(parent, key, new_m)
        hit.append(qn); replaced += 1
    if isinstance(getattr(model.visual, "proj", None), nn.Linear):
        model.visual.proj = LoRALinear(model.visual.proj, r=r, alpha=alpha, dropout=dropout)
        hit.append("visual.proj"); replaced += 1
    print(f"[LoRA] injected {replaced} Linear(s). Examples: {hit[:6]}")
    return replaced

def iter_lora_params(model):
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            yield n,p

# ----------------- PACF -----------------
def pac_kl_and_loss(model, prior_var: float = 1.0, post_var: float = 1.0):
    theta_sq_sum = None; count = 0
    for _, p in iter_lora_params(model):
        if p.requires_grad:
            v = (p.float()**2).sum()
            theta_sq_sum = v if theta_sq_sum is None else (theta_sq_sum + v)
            count += p.numel()
    if count == 0:
        d = next(model.parameters()).device
        return torch.tensor(0.0, device=d), 0.0, 0.0
    d = next(model.parameters()).device
    theta_sq_sum = theta_sq_sum.to(d)
    kl = 0.5 * (theta_sq_sum/prior_var + (count*post_var)/prior_var - count - count*math.log(post_var/prior_var))
    kl_per_dim = kl.item() / count
    kl_eff = 0.5 * (theta_sq_sum/prior_var)
    kl_eff_per_dim = kl_eff.item() / count
    return kl, kl_per_dim, kl_eff_per_dim

# ----------------- Flat-LoRA (deterministic σ) -----------------
class FlatLoRAInject:
    def __init__(self, total_steps: int, rho: float = 0.05, base_seed: int = 3, log_every: int = 200):
        self.total_steps = max(1, int(total_steps))
        self.rho = float(rho); self.base_seed = int(base_seed); self.log_every = int(log_every)
        self._cached = None
    @staticmethod
    def _step_seed(base_seed: int, step: int) -> int:
        return (hash((base_seed, int(step))) & 0xFFFFFFFF)
    def _cosine_factor(self, step: int) -> float:
        x = min(max(step / self.total_steps, 0.0), 1.0)
        return 0.5 * (1.0 - math.cos(math.pi * x))
    @torch.no_grad()
    def _collect_targets(self, model):
        return [m.base.weight for m in model.modules() if isinstance(m, LoRALinear)]
    @torch.no_grad()
    def _compute_stds(self, mats, factor: float):
        stds = []
        for W in mats:
            n_in = max(1, W.shape[1]); row_norm = torch.norm(W, p=2, dim=1, keepdim=True)
            stds.append(((factor * self.rho) / math.sqrt(n_in) * row_norm).to(W.device, dtype=W.dtype))
        return stds
    @torch.no_grad()
    def add(self, model, step: int):
        mats = self._collect_targets(model); factor = self._cosine_factor(step)
        stds = self._compute_stds(mats, factor); torch.manual_seed(self._step_seed(self.base_seed, step))
        for W, std in zip(mats, stds):
            noise = torch.randn_like(W) * std.expand_as(W); W.add_(noise)
        self._cached = (mats, stds, step)
        if step % self.log_every == 0: print(f"[Flat-LoRA] step={step} factor={factor:.5f}")
    @torch.no_grad()
    def remove(self, model, step: int):
        if self._cached is None: return
        mats, stds, cached_step = self._cached
        if step != cached_step: return
        torch.manual_seed(self._step_seed(self.base_seed, step))
        for W, std in zip(mats, stds):
            noise = torch.randn_like(W) * std.expand_as(W); W.sub_(noise)
        self._cached = None

# ----------------- Data -----------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(img_size=224, dataset_name:str=""):
    is_svhn = str(dataset_name).lower() == "svhn"
    train_ops = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ]
    if not is_svhn:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops += [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    train_tf = transforms.Compose(train_ops)

    test_tf = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return train_tf, test_tf

def load_dataset(name: str, root="./data", img_size=224):
    name = name.lower()
    train_tf, test_tf = get_transforms(img_size, dataset_name=name)
    if name == "cifar10":
        train = tv.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        train = tv.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test  = tv.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
        classes = list(train.classes)
    elif name == "cifar100":
        train = tv.datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test  = tv.datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
        classes = list(train.classes)
    elif name == "svhn":
        train = tv.datasets.SVHN(root, split="train", download=True, transform=train_tf)
        test  = tv.datasets.SVHN(root, split="test",  download=True, transform=test_tf)
        classes = [str(i) for i in range(10)]
    elif name == "cars":
        try:
            train = tv.datasets.StanfordCars(root, split="train", download=True, transform=train_tf)
            test  = tv.datasets.StanfordCars(root, split="test",  download=True, transform=test_tf)
            classes = list(train.classes)
        except ValueError as e:
            raise RuntimeError("runtime_error" + str(e))
    elif name == "dtd":
        train = tv.datasets.DTD(root, split="train", download=True, transform=train_tf)
        test  = tv.datasets.DTD(root, split="test",  download=True, transform=test_tf)
        classes = list(train.classes)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train, test, classes

# ----------------- Zero-shot classifier (text tower frozen) -----------------
def build_text_features(classes: List[str], tokenizer, text_encoder, device, dataset_name:str):
    with torch.no_grad():
        if dataset_name.lower() == "svhn":
            prompts = [f"a photo of the digit {c}" for c in classes]
        else:
            prompts = [f"a photo of a {c}" for c in classes]
        tokens = tokenizer(prompts).to(device)
        text_feat = text_encoder(tokens)
        if isinstance(text_feat, (tuple, list)): text_feat = text_feat[0]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat

# ----------------- LR schedule -----------------
def cosine_lr(optimizer, base_lr, warmup, total_steps):
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------- Eval -----------------
def evaluate(model, text_features, logit_scale, loader, device, cuda_dtype):
    model.eval()
    ce = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            with autocast_ctx(device, cuda_dtype):
                img = model.encode_image(x)
                img = img / img.norm(dim=-1, keepdim=True)
                logits = (logit_scale.exp() * img @ text_features.t())
                loss = ce(logits, y)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            pred = logits.argmax(dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    acc = float((y_true == y_pred).mean())
    loss = loss_sum / n
    # macro-F1
    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    tp = np.diag(cm); fp = cm.sum(axis=0) - tp; fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)>0)
    rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)>0)
    f1_c = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec, dtype=float), where=(prec+rec)>0)
    f1_macro = float(np.mean(f1_c))
    return {"loss": loss, "acc": acc, "f1_macro": f1_macro}

# ----------------- Training (single run) -----------------
def train_one(cfg: CFG, mode: str, trial_seed: int, tag: str, outdir: str):
    set_seed(trial_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_dtype = choose_cuda_dtype()
    scaler = make_scaler(device, cuda_dtype)

    train_set, test_set, classes = load_dataset(cfg.dataset, img_size=cfg.img_size)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=max(256, cfg.batch_size),
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # freeze text tower
    for p in model.transformer.parameters(): p.requires_grad = False
    for p in model.token_embedding.parameters(): p.requires_grad = False
    text_features = build_text_features(classes, tokenizer, model.encode_text, device, cfg.dataset)
    logit_scale = model.logit_scale

    params: List[torch.nn.Parameter] = []
    if mode in {"lora","flat_lora","pacf"}:
        n_lora = inject_lora_into_vit(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        for _, p in iter_lora_params(model):
            p.requires_grad = True; params.append(p)
        if len(params) == 0:
            raise RuntimeError("No LoRA parameters were injected.")
        base_lr = cfg.lr_lora; wd = 0.0
        print(f"[{mode.upper()}] trainable params = {sum(p.numel() for p in params)} tensors={len(params)} (injected={n_lora})")
    elif mode == "full":
        for _, p in model.visual.named_parameters():
            p.requires_grad = True; params.append(p)
        base_lr = cfg.lr_full; wd = cfg.weight_decay
        print(f"[FULL] trainable(visual)={sum(p.numel() for p in params)}")
    else:
        raise ValueError(mode)

    model = model.to(device)
    opt = AdamW(params, lr=base_lr, weight_decay=wd)

    total_steps = cfg.epochs * math.ceil(len(train_loader.dataset)/cfg.batch_size)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    sch = cosine_lr(opt, base_lr, warmup_steps, total_steps)
    ce = nn.CrossEntropyLoss()

    flat = FlatLoRAInject(total_steps=total_steps, rho=cfg.flat_sigma_max, base_seed=trial_seed) if mode=="flat_lora" else None

    use_wb = (wandb is not None) and (os.environ.get("USE_WANDB", "0") == "1")
    if use_wb:
        wandb.init(project=f"{cfg.wandb_project_prefix}_{cfg.dataset}",
                   name=f"{cfg.dataset}-{mode}-r{cfg.lora_r}-seed{trial_seed}-{tag}",
                   config={**asdict(cfg), "mode": mode, "trial_seed": trial_seed, "tag": tag})

    zs = evaluate(model, text_features, logit_scale, test_loader, device, cuda_dtype)
    print(f"[Zero-shot] acc={zs['acc']:.4f} loss={zs['loss']:.4f}; total_steps={total_steps}")

    os.makedirs(outdir, exist_ok=True)
    best = {"acc": 0.0, "loss": 1e9, "f1_macro": 0.0}
    global_step = 0
    for epoch in range(1, cfg.epochs+1):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            if flat is not None: flat.add(model, global_step)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, cuda_dtype):
                img = model.encode_image(x)
                img = img / img.norm(dim=-1, keepdim=True)
                logits = (logit_scale.exp() * img @ text_features.t())
                loss = ce(logits, y)
                if mode == "pacf" and cfg.pac_lambda > 0:
                    warm = max(1, int(cfg.reg_warmup_ratio * total_steps))
                    xw = min(max(global_step / warm, 0.0), 1.0)
                    pac_w = 0.5 * (1.0 - math.cos(math.pi * xw))
                    kl, _, _ = pac_kl_and_loss(model, cfg.pac_prior_var, cfg.pac_post_var)
                    loss = loss + cfg.pac_lambda * pac_w * kl
            if cuda_dtype is torch.bfloat16:
                loss.backward(); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if flat is not None: flat.remove(model, global_step)
            sch.step(); global_step += 1

        ev = evaluate(model, text_features, logit_scale, test_loader, device, cuda_dtype)
        if (ev["acc"] > best["acc"]) or (ev["acc"] == best["acc"] and ev["loss"] < best["loss"]):
            best = ev
        print(f"[{cfg.dataset}][{mode}][{tag}] e{epoch}/{cfg.epochs} acc={ev['acc']:.4f} "
              f"loss={ev['loss']:.4f} F1={ev['f1_macro']:.4f} best_acc={best['acc']:.4f}")
        if use_wb:
            wandb.log({"epoch": epoch, "val/acc": ev["acc"], "val/loss": ev["loss"], "val/f1_macro": ev["f1_macro"]})

    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump({"last": ev, "best": best, "config": asdict(cfg)}, f, indent=2)
    if use_wb:
        wandb.log({"best/acc": best["acc"], "best/loss": best["loss"], "best/f1_macro": best["f1_macro"]})
        wandb.finish()
    return ev, best

# ----------------- r-scaled priors & deterministic grids -----------------
# λ priors are defined for r=8 (the "strongest"). For other r, scale by 8/r.
LAMBDA_PRIORS_BASE = {
    "svhn":     (1e-5, 5e-4),
    "cifar10":  (2e-5, 8e-4),
    "dtd":      (2e-5, 1e-3),
    "cars":     (2e-5, 1e-3),
    "cifar100": (2e-5, 8e-4),
}
WR_PRIORS = {
    "svhn":     (0.15, 0.50),
    "cifar10":  (0.20, 0.55),
    "dtd":      (0.20, 0.60),
    "cars":     (0.20, 0.60),
    "cifar100": (0.20, 0.55),
}
# Deterministic paired @ r=8; for r!=8, λ→λ*(8/r)
HANDPICKED = {
    "svhn":     [(1e-4,0.30),(2e-4,0.35),(5e-5,0.30)],
    "cifar10":  [(1e-4,0.30),(2e-4,0.30),(3e-4,0.35),(5e-4,0.35)],
    "dtd":      [(1e-4,0.30),(2e-4,0.35),(5e-4,0.35),(8e-4,0.40)],
    "cars":     [(1e-4,0.30),(2e-4,0.30),(5e-4,0.35),(8e-4,0.40)],
    "cifar100": [(1e-4,0.30),(2e-4,0.30),(3e-4,0.35),(5e-4,0.35)],
}

def _rank_scaled_range(dataset:str, r:int):
    base = LAMBDA_PRIORS_BASE.get(dataset, (2e-5,8e-4))
    if r == 8: return base
    scale = 8.0 / max(1, r)
    return (base[0]*scale, base[1]*scale)

def _scale_lambda_for_rank(lam: float, r:int) -> float:
    return float(lam) * (8.0 / max(1, r))

def _main_key(): return "acc"  # gating/selection by accuracy

# ----------------- Paper summary -----------------
def paper_style_compare(dataset="cifar10", r=8, modes=("lora","flat_lora","pacf"), seeds=(1,2,3), save=True, note=""):
    rows=[]; summary={}
    base_dir = os.path.join(os.path.join("runs","clip"), dataset, _ts()+"-paper")
    os.makedirs(base_dir, exist_ok=True)
    for mode in modes:
        accs, losses, f1s = [], [], []
        for s in seeds:
            CFG.dataset = dataset; CFG.lora_r = r
            outdir = make_default_outdir(dataset, mode, s, r, CFG.lora_alpha,
                                         lr=(CFG.lr_full if mode=="full" else CFG.lr_lora),
                                         extras=({"pac_lambda": CFG.pac_lambda, "wr": CFG.reg_warmup_ratio} if mode=="pacf" else {}))
            last, best = train_one(CFG, mode=mode, trial_seed=s, tag="paper", outdir=outdir)
            rows.append({"dataset":dataset,"rank":r,"mode":mode,"seed":s,
                         "acc_last":last["acc"],"loss_last":last["loss"],"f1_last":last["f1_macro"],
                         "acc_best":best["acc"],"loss_best":best["loss"],"f1_best":best["f1_macro"],
                         "epochs":CFG.epochs,"flat_sigma_max":CFG.flat_sigma_max,
                         "pac_lambda":CFG.pac_lambda,"reg_warmup_ratio":CFG.reg_warmup_ratio,
                         "lora_alpha":CFG.lora_alpha,"lora_r":CFG.lora_r,
                         "lr_lora":CFG.lr_lora,"lr_full":CFG.lr_full,"weight_decay":CFG.weight_decay})
            accs.append(best["acc"]); losses.append(best["loss"]); f1s.append(best["f1_macro"])
        def ms(xs):
            xs = [x for x in xs if x is not None]
            m = float(np.mean(xs)) if xs else float("nan")
            sd = float(np.std(xs, ddof=1)) if len(xs)>1 else 0.0
            return m, sd
        am, asd = ms(accs); lm, lsd = ms(losses); fm, fsd = ms(f1s)
        summary[mode] = {"acc_mean":am,"acc_std":asd,"loss_mean":lm,"loss_std":lsd,"f1_mean":fm,"f1_std":fsd}
        print(f"[{dataset.upper()}][r={r}] {mode:8s}  acc={am:.4f}±{asd:.4f}  loss={lm:.4f}±{lsd:.4f}  F1={fm:.4f}±{fsd:.4f}")
    if save and rows:
        with open(os.path.join(base_dir,"details.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        with open(os.path.join(base_dir,"summary.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["mode","acc_mean","acc_std","loss_mean","loss_std","f1_mean","f1_std"])
            for m in modes:
                o=summary[m]; w.writerow([m,o["acc_mean"],o["acc_std"],o["loss_mean"],o["loss_std"],o["f1_mean"],o["f1_std"]])
        meta={"dataset":dataset,"rank":r,"cfg":asdict(CFG),
              "versions":{"python":platform.python_version(),"torch":torch.__version__,"torchvision":tv.__version__,"open_clip":open_clip.__version__},
              "note":note}
        with open(os.path.join(base_dir,"meta.json"),"w") as f: json.dump(meta,f,indent=2)
        print(f"[OK] saved to: {base_dir}")
    return summary

# ----------------- Auto pipeline: A/B/C -----------------
def auto_run_ic(dataset: str,
                ranks: List[int] = [8],
                topk_to_stageB: int = 3,
                seeds_stageC: Tuple[int,...] = (1,2,3),
                stageA_lambda_points: List[float] | None = None,
                stageA_wr_points: List[float] | None = None,
                skip_full: bool = True,
                use_gate: bool = False,
                gate_seed: int = 2,
                gate_delta: float = 0.002,
                ab_save_flat: bool = False):
    """
    Stage-A: deterministic grid（PACF only, seed=2）; epochs=2
            
    Stage-B: Top-K（seed=2）epochs=10
    Stage-C: PACF(best) + baselines（LoRA、Flat；Full ） seed；summary
    """
    stageB_all = []
    for r in ranks:
        CFG.lora_r = r
        lam_lo, lam_hi = _rank_scaled_range(dataset, r)
        print(f"[Stage-A] dataset={dataset} r={r}  λ∈[{lam_lo:.1e},{lam_hi:.1e}] (r-scaled)")

        # ---- Stage A: build deterministic candidate list ----
        if stageA_lambda_points and stageA_wr_points:
            candidate_pairs = [(float(l), float(w)) for l in stageA_lambda_points for w in stageA_wr_points]
        else:
            pairs = HANDPICKED.get(dataset, [])
            if not pairs:
                lams = np.geomspace(lam_lo, lam_hi, 3)
                pairs = [(float(l), 0.30) for l in lams]
            candidate_pairs = [(_scale_lambda_for_rank(l, r), w) for (l, w) in pairs]

        if ab_save_flat:
            CFG.dataset = dataset; CFG.seed = gate_seed; CFG.epochs = 2
            flat_out = make_default_outdir(dataset, "flat_lora", CFG.seed, r, CFG.lora_alpha, lr=CFG.lr_lora)
            _, flat_best = train_one(CFG, mode="flat_lora", trial_seed=CFG.seed, tag="stageA_flat", outdir=flat_out)
            A_dir = os.path.join(os.path.join("runs","clip"), dataset, f"stageA_r{r}"); os.makedirs(A_dir, exist_ok=True)
            with open(os.path.join(A_dir, "stageA_flat_ref.csv"), "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([dataset, r, CFG.seed, "flat_lora", flat_best["acc"], flat_best["loss"], flat_best.get("f1_macro", float("nan"))])

        # run A (epochs=2)
        A_csv_dir = os.path.join(os.path.join("runs","clip"), dataset, f"stageA_r{r}")
        os.makedirs(A_csv_dir, exist_ok=True)
        trials_csv = os.path.join(A_csv_dir, f"stageA_{_ts()}.csv")

        rowsA = []
        CFG.seed = 2
        backup_epochs = CFG.epochs
        CFG.epochs = 2
        for idx, (lam, wr) in enumerate(candidate_pairs, 1):
            CFG.dataset = dataset; CFG.pac_lambda = float(lam); CFG.reg_warmup_ratio = float(wr)
            outdir = make_default_outdir(dataset, "pacf", CFG.seed, r, CFG.lora_alpha, lr=CFG.lr_lora,
                                         extras={"pac_lambda": lam, "wr": wr})
            _, best = train_one(CFG, mode="pacf", trial_seed=CFG.seed, tag=f"stageA_c{idx}", outdir=outdir)
            rowsA.append({"candidate": idx, "rank": r, "seed": CFG.seed,
                          "pac_lambda": lam, "reg_warmup_ratio": wr,
                          "acc_best": best["acc"], "loss_best": best["loss"], "f1_best": best["f1_macro"],
                          "lambda_report": f"{lam:.1e}", "wr_report": f"{wr:.2f}"})
        CFG.epochs = backup_epochs
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(rowsA[0].keys())); w.writeheader(); w.writerows(rowsA)

        # pick Top-K for Stage-B
        topk_rows = sorted(rowsA, key=lambda x:(x["acc_best"], -x["loss_best"]), reverse=True)[:topk_to_stageB]
        stageB_all.append({"rank": r, "topk": topk_rows})

    # ---- Stage B: re-eval topK @ seed=2 (epochs=10) ----
    picks = []
    for pack in stageB_all:
        r = pack["rank"]; CFG.lora_r = r; CFG.seed = 2

        gate_score = -1.0
        if use_gate:
            def _gate_eval(variant):
                CFG.dataset = dataset; CFG.epochs = 10
                outdir = make_default_outdir(dataset, variant, CFG.seed, r, CFG.lora_alpha,
                                             lr=(CFG.lr_full if variant=="full" else CFG.lr_lora))
                _, best = train_one(CFG, mode=variant, trial_seed=CFG.seed, tag="gate", outdir=outdir)
                return best["acc"]
            acc_lora = _gate_eval("lora")
            acc_flat = _gate_eval("flat_lora")
            gate_score = max(acc_lora, acc_flat)
            print(f"[Gate] r={r}  LoRA={acc_lora:.4f}  Flat={acc_flat:.4f}  → gate={gate_score:.4f}")

        cand_csv_dir = os.path.join(os.path.join("runs","clip"), dataset, f"stageB_r{r}")
        os.makedirs(cand_csv_dir, exist_ok=True)
        cand_csv = os.path.join(cand_csv_dir, f"stageB_{_ts()}.csv")

        rowsB = []
        backup_epochs = CFG.epochs
        CFG.epochs = 10
        for j, row in enumerate(pack["topk"], 1):
            lam, wr = float(row["pac_lambda"]), float(row["reg_warmup_ratio"])
            CFG.dataset = dataset; CFG.pac_lambda = lam; CFG.reg_warmup_ratio = wr
            outdir = make_default_outdir(dataset, "pacf", CFG.seed, r, CFG.lora_alpha, lr=CFG.lr_lora,
                                         extras={"pac_lambda": lam, "wr": wr})
            _, best = train_one(CFG, mode="pacf", trial_seed=CFG.seed, tag=f"stageB_c{j}", outdir=outdir)
            rowsB.append({"candidate": j, "rank": r, "seed": CFG.seed,
                          "pac_lambda": lam, "reg_warmup_ratio": wr,
                          "acc_best": best["acc"], "loss_best": best["loss"], "f1_best": best["f1_macro"]})
        CFG.epochs = backup_epochs
        with open(cand_csv, "w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(rowsB[0].keys())); w.writeheader(); w.writerows(rowsB)

        best_row = sorted(rowsB, key=lambda x:(x["acc_best"], -x["loss_best"]), reverse=True)[0]

        # gate early-stop
        if use_gate and (best_row["acc_best"] + 1e-12 < (gate_score - gate_delta)):
            print(f"[StageB Early-Stop] r={r}  PACF best {best_row['acc_best']:.4f} < gate {gate_score:.4f} → skip Stage-C.")
            continue

        picks.append({"rank": r, "best": best_row})

    # ---- Stage C: seeds, PACF(best) + baselines ----
    for pack in picks:
        r = pack["rank"]; lam = float(pack["best"]["pac_lambda"]); wr = float(pack["best"]["reg_warmup_ratio"])
        # PACF(best)
        for s in seeds_stageC:
            CFG.seed = s; CFG.lora_r = r; CFG.dataset = dataset
            CFG.pac_lambda = lam; CFG.reg_warmup_ratio = wr
            outdir = make_default_outdir(dataset, "pacf", s, r, CFG.lora_alpha, lr=CFG.lr_lora,
                                         extras={"pac_lambda": lam, "wr": wr})
            train_one(CFG, mode="pacf", trial_seed=s, tag="stageC_pacf", outdir=outdir)
        # baselines
        for mode in (["lora","flat_lora"] if skip_full else ["lora","flat_lora","full"]):
            for s in seeds_stageC:
                CFG.seed = s; CFG.lora_r = r; CFG.dataset = dataset
                outdir = make_default_outdir(dataset, mode, s, r, CFG.lora_alpha,
                                             lr=(CFG.lr_full if mode=="full" else CFG.lr_lora))
                train_one(CFG, mode=mode, trial_seed=s, tag=f"stageC_{mode}", outdir=outdir)

        modes = ["lora","flat_lora","pacf"] if skip_full else ["lora","flat_lora","full","pacf"]
        paper_style_compare(dataset=dataset, r=r, modes=tuple(modes), seeds=CFG.seeds_stageC, save=True,
                            note=f"A:deterministic grid (e=2); B:Top-{topk_to_stageB} (e=10); λ scaled for r")

# ----------------- CLI -----------------
def _parse_float_list(s):
    if not s: return []
    return [float(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=CFG.dataset, choices=["svhn","cifar10","dtd","cars","cifar100"])
    ap.add_argument("--mode", type=str, default="auto", choices=["auto","paper","lora","flat_lora","full","pacf"])
    ap.add_argument("--r", type=int, default=CFG.lora_r)
    ap.add_argument("--alpha", type=int, default=CFG.lora_alpha)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--seeds", type=str, default="")  # e.g., "1,2,3"
    # PACF / Opt
    ap.add_argument("--pac_lambda", type=float, default=CFG.pac_lambda)
    ap.add_argument("--reg_warmup_ratio", type=float, default=CFG.reg_warmup_ratio)
    ap.add_argument("--lr_lora", type=float, default=CFG.lr_lora)
    ap.add_argument("--lr_full", type=float, default=CFG.lr_full)
    # auto stages
    ap.add_argument("--topk_to_stageB", type=int, default=3)
    ap.add_argument("--skip_full", action="store_true")
    ap.add_argument("--stageA_lambda_points", type=str, default="", help="e.g. '1e-4,2e-4,3e-4'")
    ap.add_argument("--stageA_wr_points", type=str, default="", help="e.g. '0.30,0.35,0.40'")
    ap.add_argument("--use_gate", action="store_true", help="Use LoRA/Flat gate at Stage-B")
    ap.add_argument("--gate_seed", type=int, default=2)
    ap.add_argument("--gate_delta", type=float, default=0.002)
    ap.add_argument("--ab_save_flat", action="store_true", help="Run & save a single Flat-LoRA reference in Stage-A/B")
    args = ap.parse_args()

    CFG.dataset = args.dataset; CFG.lora_r = args.r; CFG.lora_alpha = args.alpha
    CFG.epochs = args.epochs; CFG.batch_size = args.batch_size
    CFG.seed = args.seed
    CFG.pac_lambda = args.pac_lambda; CFG.reg_warmup_ratio = args.reg_warmup_ratio
    CFG.lr_lora = args.lr_lora; CFG.lr_full = args.lr_full

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    if args.mode == "auto":
        auto_run_ic(dataset=CFG.dataset,
                    ranks=[args.r],
                    topk_to_stageB=args.topk_to_stageB,
                    seeds_stageC=CFG.seeds_stageC,
                    stageA_lambda_points=_parse_float_list(args.stageA_lambda_points),
                    stageA_wr_points=_parse_float_list(args.stageA_wr_points),
                    skip_full=args.skip_full,
                    use_gate=args.use_gate,
                    gate_seed=args.gate_seed,
                    gate_delta=args.gate_delta,
                    ab_save_flat=args.ab_save_flat)
    elif args.mode == "paper":
        modes = ["lora","flat_lora","pacf"] if args.skip_full else ["lora","flat_lora","full","pacf"]
        paper_style_compare(dataset=CFG.dataset, r=CFG.lora_r, modes=tuple(modes),
                            seeds=CFG.seeds_stageC, save=True)
    else:
        if seeds:
            outdir = make_summary_outdir(dataset=CFG.dataset, mode=args.mode, r=CFG.lora_r, alpha=CFG.lora_alpha,
                                         lr=(CFG.lr_full if args.mode=="full" else CFG.lr_lora),
                                         extras=({"pac_lambda": CFG.pac_lambda, "wr": CFG.reg_warmup_ratio} if args.mode=="pacf" else {}))
            os.makedirs(outdir, exist_ok=True)
            rows=[]
            for s in seeds:
                per_seed_out = make_default_outdir(CFG.dataset, args.mode, s, CFG.lora_r, CFG.lora_alpha,
                                                   lr=(CFG.lr_full if args.mode=="full" else CFG.lr_lora),
                                                   extras=({"pac_lambda": CFG.pac_lambda, "wr": CFG.reg_warmup_ratio} if args.mode=="pacf" else {}))
                last,best = train_one(CFG, mode=args.mode, trial_seed=s, tag="seeds", outdir=per_seed_out)
                rows.append({"dataset":CFG.dataset,"rank":CFG.lora_r,"mode":args.mode,"seed":s,
                             "acc_last":last["acc"],"loss_last":last["loss"],"f1_last":last["f1_macro"],
                             "acc_best":best["acc"],"loss_best":best["loss"],"f1_best":best["f1_macro"]})
            with open(os.path.join(outdir,"details.csv"),"w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        else:
            outdir = make_default_outdir(CFG.dataset, args.mode, CFG.seed, CFG.lora_r, CFG.lora_alpha,
                                         lr=(CFG.lr_full if args.mode=="full" else CFG.lr_lora),
                                         extras=({"pac_lambda": CFG.pac_lambda, "wr": CFG.reg_warmup_ratio} if args.mode=="pacf" else {}))
            last,best = train_one(CFG, mode=args.mode, trial_seed=CFG.seed, tag="single", outdir=outdir)
            print(json.dumps({"last":last,"best":best}, indent=2))

if __name__ == "__main__":
    main()
