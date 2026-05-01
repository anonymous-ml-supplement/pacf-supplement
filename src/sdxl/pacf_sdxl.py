#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACF + Flat-LoRA + DreamBooth for SDXL

This is a runnable implementation for the qualitative SDXL personalization artifact.
The SDXL artifact is qualitative unless quantitative metrics are provided separately.

Features:
- DreamBooth-style personalization (instance images only, no class data by default)
- LoRA / Flat-LoRA / PACF variants
- Deterministic Stage A/B/C protocol (grid -> select -> multi-seed)
- Optional CLIPScore-based evaluation and sampling
- Optional Weights & Biases logging (no hard-coded entity)

Notes:
- No user-identifying information is included in this file.
- Paths are relative; pass explicit `--output_dir` for outputs.
"""

import os as _os
_os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import os, math, time, json, random, argparse, csv, gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed as acc_set_seed

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import make_image_grid

# --------------- optional W&B ------------------
try:
    import wandb as _wandb_mod
    _WANDB = hasattr(_wandb_mod, "init")
    wandb = _wandb_mod
except Exception:
    wandb = None
    _WANDB = False


# =========================================================
# Utils
# =========================================================
def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    acc_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CFG:
    pretrained: str = "stabilityai/stable-diffusion-xl-base-1.0"
    image_size: int = 1024

    # LoRA
    rank: int = 4
    alpha: int = 4
    lora_scope: str = "unet"

    # Train
    steps: int = 500
    batch: int = 1
    lr: float = 1e-4
    grad_accum: int = 1
    log_every: int = 100

    # Flat
    flat_sigma: float = 0.10

    # PACF
    pac_lambda: float = 1e-5
    reg_warmup_ratio: float = 0.25
    pacf_gamma: float = 0.05
    
    # Eval
    eval_steps: int = 35
    eval_guidance: float = 6.0
    eval_prompts: Tuple[str, ...] = (
        "a TOK icon of a flying bird, in the style of TOK",
    )

    # Infra
    seed: int = 2
    mixed_precision: str = "bf16"
    mem_eff_attn: bool = True

    # DreamBooth instance/class directories
    default_instance_dir: str = "images_instance"
    default_class_dir: str = "images_class"

CFG = CFG()


# =========================================================
# Dataset
# =========================================================
class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, size: int = 1024):
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            paths += list(root.glob(ext))
        self.paths = sorted(paths)
        if not self.paths:
            raise ValueError(f"No images found in {root}")
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize(
            (self.size, self.size), Image.BICUBIC
        )
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr * 2 - 1).transpose(2, 0, 1)
        return torch.from_numpy(arr)


# =========================================================
# LoRA attach (PEFT)
# =========================================================
def add_lora_to_unet_and_textenc(pipeline, rank=8, alpha=16, lora_scope="unet"):
    trainable = []
    try:
        from peft import LoraConfig
        try:
            from peft import TaskType
            peft_cfg = LoraConfig(
                r=int(rank),
                lora_alpha=int(alpha),
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none",
                task_type=getattr(TaskType, "UNET_TUNING", None),
            )
        except Exception:
            peft_cfg = LoraConfig(
                r=int(rank),
                lora_alpha=int(alpha),
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none",
            )

        if not hasattr(pipeline.unet, "add_adapter"):
            raise RuntimeError("UNet.add_adapter not found; update diffusers >= 0.27.x")

        pipeline.unet.add_adapter(peft_cfg, adapter_name="default")
        if hasattr(pipeline.unet, "set_adapter"):
            pipeline.unet.set_adapter("default")

        for n, p in pipeline.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
                trainable.append(p)

        if not trainable:
            raise RuntimeError("add_adapter succeeded but no LoRA params found.")

        return trainable

    except Exception as e:
        raise RuntimeError(f"LoRA attach failed: {e}")


def iter_lora_params(mods: List[nn.Module]):
    for m in mods:
        if m is None:
            continue
        for n, p in m.named_parameters():
            if p.requires_grad and (
                "lora" in n or
                "adapter" in n or
                n.endswith("lora_A.weight") or
                n.endswith("lora_B.weight")
            ):
                yield n, p


# =========================================================
# SDXL Helper: prompt encoding + time ids
# =========================================================
def sdxl_encode_prompts(pipe, prompt: str, device, do_cfg=False, num_images_per_prompt=1):
    out = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_cfg,
    )
    if isinstance(out, tuple):
        if do_cfg:
            if len(out) >= 4:
                pe, ne, pooled, npooled = out[:4]
                return pe, pooled, ne, npooled
            pe, pooled = out[:2]
            return pe, pooled, None, None
        else:
            if len(out) == 2:
                pe, pooled = out
                return pe, pooled, None, None
            elif len(out) >= 4:
                pe, _, pooled, _ = out[:4]
                return pe, pooled, None, None
    return out, None, None, None


def sdxl_time_ids(pipe, img_size: int, device, batch_size: int):
    original_size = (img_size, img_size)
    target_size = (img_size, img_size)
    crops = (0, 0)
    dtype = getattr(getattr(pipe, "text_encoder", None), "dtype", None) or pipe.unet.dtype
    proj_dim = 0
    te2 = getattr(pipe, "text_encoder_2", None)
    if te2 is not None:
        pd = getattr(getattr(te2, "config", None), "projection_dim", None)
        if isinstance(pd, int):
            proj_dim = pd

    add = None
    try:
        add = pipe._get_add_time_ids(
            original_size=original_size,
            crops_coords_top_left=crops,
            target_size=target_size,
            dtype=dtype,
            text_encoder_projection_dim=proj_dim,
        )
    except TypeError:
        add = None

    if add is None:
        try:
            add = pipe._get_add_time_ids(
                original_size=original_size,
                crops_coords_top_left=crops,
                target_size=target_size,
                dtype=dtype,
                text_encoder_projection_dim=proj_dim,
                device=device,
            )
        except TypeError:
            add = None

    if add is None:
        add = pipe._get_add_time_ids(
            original_size=original_size,
            crops_coords_top_left=crops,
            target_size=target_size,
            dtype=dtype,
        )
    add = add.to(device)
    return add.repeat(batch_size, 1)

# =========================================================
# Flat-LoRA + PACF
# =========================================================
def _flat_sigma_factor(step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    x = min(max(step / float(total_steps), 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def perturb_once_lora_params(
    modules: List[nn.Module],
    base_sigma: float,
    global_step: int,
    total_steps: int,
    base_seed: int = 3,
):
    touched = []
    if base_sigma <= 0:
        return touched

    factor = _flat_sigma_factor(global_step, total_steps)
    rho = float(base_sigma) * factor
    if rho <= 0:
        return touched

    torch.manual_seed((hash((int(base_seed), int(global_step))) & 0xFFFFFFFF))

    with torch.no_grad():
        for _, p in iter_lora_params(modules):
            W = p.data
            orig = W.clone()

            W2 = W.view(W.shape[0], -1)
            n_in = max(1, W2.shape[1])
            row_norm = W2.norm(p=2, dim=1, keepdim=True)
            std = (rho / math.sqrt(n_in)) * row_norm
            std = std.expand_as(W2).view_as(W)

            noise = torch.randn_like(W) * std
            W.add_(noise)

            touched.append((p, orig))

    return touched


def revert_perturb(touched):
    with torch.no_grad():
        for p, orig in touched:
            p.data.copy_(orig)


def pacf_loss(unet, latents, timesteps, cond_embed, added_cond_kwargs, scheduler, same_t=False):
    bsz = latents.shape[0]
    device = latents.device
    t1 = timesteps
    t2 = timesteps if same_t else torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (bsz,),
        device=device,
    )

    noise1 = torch.randn_like(latents)
    noise2 = torch.randn_like(latents)
    v1 = scheduler.add_noise(latents, noise1, t1)
    v2 = scheduler.add_noise(latents, noise2, t2)

    e1 = unet(
        v1,
        t1,
        encoder_hidden_states=cond_embed,
        added_cond_kwargs=added_cond_kwargs,
    ).sample
    e2 = unet(
        v2,
        t2,
        encoder_hidden_states=cond_embed,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    return F.mse_loss(e1.float(), e2.float())
def pacf_student_forward_same_x(
    unet,
    model_in,
    ts,
    encs,
    added_cond_kwargs_cfg,
    modules_for_lora,
    pacf_gamma: float,
    global_step: int,
    total_steps: int,
    pacf_seed: int = 7,
):
    if pacf_gamma <= 0:
        return None, []

    touched = perturb_once_lora_params(
        modules_for_lora,
        base_sigma=pacf_gamma,
        global_step=global_step,
        total_steps=total_steps,
        base_seed=pacf_seed,
    )
    pred_pert = unet(
        model_in,
        ts,
        encoder_hidden_states=encs,
        added_cond_kwargs=added_cond_kwargs_cfg,
    ).sample
    return pred_pert, touched


# =========================================================
# Eval & CLIPScore
# =========================================================
@dataclass
class EvalPrompt:
    text: str
    seed: int
    guidance: float
    steps: int


def sample_and_grid(
    pipe: StableDiffusionXLPipeline,
    prompts: List[EvalPrompt],
    outdir: Path,
    title: str,
):
    use_dtype = getattr(pipe.unet, "dtype", torch.float32)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    use_autocast = use_dtype in (torch.bfloat16, torch.float16)

    ims = []
    ctx = torch.autocast(
        device_type=device_type,
        dtype=use_dtype,
        enabled=use_autocast,
    )
    with ctx:
        for ep in prompts:
            g = torch.Generator(device=pipe.device).manual_seed(ep.seed)
            im = pipe(
                ep.text,
                guidance_scale=ep.guidance,
                num_inference_steps=ep.steps,
                generator=g,
            ).images[0]
            ims.append(im)

    grid = make_image_grid(ims, rows=1, cols=len(prompts))
    ensure_dir(outdir)
    p = outdir / f"{title}.png"
    grid.save(p)
    return p


def try_clip_score(image_path: Path, texts: List[str]) -> float:
    try:
        from transformers import CLIPProcessor, CLIPModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        image = Image.open(image_path).convert("RGB")
        scores = []
        for t in texts:
            inputs = processor(
                text=[t],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits_per_image
                scores.append(float(logits.squeeze().item()))
        return float(np.mean(scores))
    except Exception:
        return 0.0


# =========================================================
# Build SDXL pipeline
# =========================================================
def build_pipe(pretrained: str, dtype, device, mem_eff=True) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained,
        torch_dtype=dtype,
        add_watermarker=False,
        use_safetensors=True,
    )
    if mem_eff:
        try:
            pipe.enable_vae_slicing()
            pipe.unet.enable_forward_chunking()
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe



# =========================================================
# Generic training for one stage (non-DreamBooth; single prompt)
# =========================================================
def train_one_stage(
    accelerator: Accelerator,
    stage_name: str,
    pipe: StableDiffusionXLPipeline,
    dataloader: DataLoader,
    steps: int,
    lr: float,
    grad_accum: int,
    flat_sigma: float,
    flat_seed: int,
    use_pacf: bool,
    pac_lambda: float,
    reg_warmup_ratio: float,
    scheduler_train: DDPMScheduler,
    log_every: int,
    sample_prompts: List[EvalPrompt],
    outdir: Path,
    project: Optional[str],
    instance_prompt: str,
    pacf_gamma: float,
):
    device = accelerator.device
    img_size = CFG.image_size

    # optimizer for LoRA only
    lora_params = [
        p for n, p in pipe.unet.named_parameters()
        if p.requires_grad and "lora" in n
    ]
    optim = torch.optim.AdamW(lora_params, lr=lr)
    lr_sched = get_scheduler(
        "cosine",
        optimizer=optim,
        num_warmup_steps=max(1, int(0.03 * steps)),
        num_training_steps=steps,
    )

    pipe.unet.train()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.train()
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.train()

    # encode cond / uncond prompt once
    with torch.no_grad():
        cond_embed, cond_pooled, _, _ = sdxl_encode_prompts(
            pipe,
            instance_prompt,
            device,
            do_cfg=False,
        )
        uncond_embed, uncond_pooled, _, _ = sdxl_encode_prompts(
            pipe,
            "",
            device,
            do_cfg=False,
        )

    pipe.unet, optim, lr_sched, dataloader = accelerator.prepare(
        pipe.unet,
        optim,
        lr_sched,
        dataloader,
    )

    global_step = 0
    micro_step = 0

    while global_step < steps:
        for images in dataloader:
            if global_step >= steps:
                break

            images = images.to(device=device, dtype=pipe.unet.dtype)
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            bsz = latents.shape[0]
            t = torch.randint(
                0,
                scheduler_train.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(latents)
            noisy = scheduler_train.add_noise(latents, noise, t)

            time_ids = sdxl_time_ids(pipe, img_size, device, bsz)

            # prepare cond / uncond batch
            cond_embed_b = cond_embed.repeat(bsz, 1, 1)
            cond_pooled_b = cond_pooled.repeat(bsz, 1)
            uncond_embed_b = uncond_embed.repeat(bsz, 1, 1)
            uncond_pooled_b = uncond_pooled.repeat(bsz, 1)

            added_cond_kwargs_cfg = {
                "text_embeds": torch.cat(
                    [uncond_pooled_b, cond_pooled_b],
                    dim=0,
                ),
                "time_ids": torch.cat(
                    [time_ids, time_ids],
                    dim=0,
                ),
            }
            added_cond_kwargs_pacf = {
                "text_embeds": cond_pooled_b,
                "time_ids": time_ids,
            }

            model_in = torch.cat([noisy, noisy], dim=0)
            ts = torch.cat([t, t], dim=0)
            encs = torch.cat(
                [uncond_embed_b, cond_embed_b],
                dim=0,
            )
            target = torch.cat([noise, noise], dim=0)

            # ---------------------------------------------------------
            # Flat-LoRA: perturb BEFORE base loss (keeps your current behavior)
            # PACF: base loss stays clean; perturb only inside PACF regularizer
            # ---------------------------------------------------------
                        # ---------------------------------------------------------
            # Flat-LoRA: perturb BEFORE base loss (keeps your current behavior)
            # PACF: base loss stays clean; perturb only inside PACF regularizer
            # ---------------------------------------------------------
            touched_flat = []
            if (not use_pacf) and (flat_sigma > 0):
                touched_flat = perturb_once_lora_params(
                    [pipe.unet],
                    base_sigma=flat_sigma,
                    global_step=global_step,
                    total_steps=steps,
                    base_seed=flat_seed,
                )
            
            # ---------- base diffusion forward (clean for LoRA/PACF, perturbed for Flat) ----------
            pred = pipe.unet(
                model_in,
                ts,
                encoder_hidden_states=encs,
                added_cond_kwargs=added_cond_kwargs_cfg,
            ).sample
            
            loss_simple = F.mse_loss(pred.float(), target.float())
            
            # backward base part first (so clean graph finishes before any PACF perturb)
            accelerator.backward((loss_simple / grad_accum).float())
            
            revert_perturb(touched_flat)
            
            # ---------- PACF part (one extra forward) ----------
            if use_pacf and (pac_lambda > 0) and (pacf_gamma > 0):
                warm = min(1.0, max(0.0, global_step / max(1, int(reg_warmup_ratio * steps))))
                teacher = pred.detach()  # stopgrad teacher from base forward
            
                pred_pert, touched_pacf = pacf_student_forward_same_x(
                    unet=pipe.unet,
                    model_in=model_in,
                    ts=ts,
                    encs=encs,
                    added_cond_kwargs_cfg=added_cond_kwargs_cfg,
                    modules_for_lora=[pipe.unet],
                    pacf_gamma=pacf_gamma,
                    global_step=global_step,
                    total_steps=steps,
                    pacf_seed=flat_seed + 999,
                )
                try:
                    pac_v = F.mse_loss(pred_pert.float(), teacher.float())
                    accelerator.backward(((pac_lambda * warm * pac_v) / grad_accum).float())
                finally:
                    revert_perturb(touched_pacf)

            else:
                pac_v = torch.tensor(0.0, device=device, dtype=loss_simple.dtype)
                warm = 0.0
            
            micro_step += 1




            if micro_step % grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                lr_sched.step()
                global_step += 1

                if accelerator.is_main_process and (global_step % log_every == 0):
                    if _WANDB and project:
                        wandb.log({
                            f"{stage_name}/step": int(global_step),
                            f"{stage_name}/loss_simple": float(loss_simple.detach().item()),
                            f"{stage_name}/loss_pacf": float(pac_v.detach().item()),
                            f"{stage_name}/pacf_warm": float(warm),
                            f"{stage_name}/lr": float(lr_sched.get_last_lr()[0]),
                        })
                    grid_path = sample_and_grid(
                        pipe,
                        sample_prompts,
                        outdir,
                        f"{stage_name}_step{global_step}",
                    )
                    if _WANDB and project:
                        wandb.log({
                            f"{stage_name}/samples": wandb.Image(str(grid_path)),
                        })

    if accelerator.is_main_process:
        ldir = outdir / f"lora_{stage_name}"
        ensure_dir(ldir)
        unet_to_save = accelerator.unwrap_model(pipe.unet)
        saved = False

        if hasattr(pipe, "save_lora_weights"):
            try:
                pipe.save_lora_weights(
                    ldir,
                    weight_name="pytorch_lora_weights.safetensors",
                    unet_lora_layers=pipe.unet,
                )
                saved = True
            except Exception as e:
                print("[WARN] save_lora_weights failed; fallback to save_attn_procs:", e)

        if (not saved) and hasattr(unet_to_save, "save_attn_procs"):
            unet_to_save.save_attn_procs(ldir)
            saved = True

        if not saved:
            raise RuntimeError("Cannot save LoRA weights; missing save_lora_weights/save_attn_procs.")

        with open(ldir / "hparams.json", "w") as f:
            json.dump({
                "stage": stage_name,
                "steps": steps,
                "lr": lr,
                "flat_sigma": float(flat_sigma),
                "use_pacf": bool(use_pacf),
                "pac_lambda": float(pac_lambda),
                "reg_warmup_ratio": float(reg_warmup_ratio),
                "pacf_gamma": float(pacf_gamma),
            }, f, indent=2)

        if _WANDB and project:
            wandb.save(str(ldir / "hparams.json"))


# =========================================================
# CSV helper
# =========================================================
def append_score_csv(outdir: Path, row: dict, filename: str = "scores.csv"):
    csv_path = outdir / filename
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # data / model
    ap.add_argument("--pretrained", type=str, default=CFG.pretrained)
    ap.add_argument("--instance_data", type=str, default=CFG.default_instance_dir)
    ap.add_argument("--instance_prompt", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=CFG.image_size)

    # stage / mode / seeds / variant
    ap.add_argument("--stage", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "single", "seeds"])
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--variant", type=str, default="lora", choices=["lora", "flat", "pacf"])
    ap.add_argument("--topk_to_stageB", type=int, default=2)

    # LoRA
    ap.add_argument("--rank", type=int, default=CFG.rank)
    ap.add_argument("--alpha", type=int, default=CFG.alpha)
    ap.add_argument("--lora_scope", type=str, default=CFG.lora_scope, choices=["unet", "both"])

    # train
    ap.add_argument("--steps", type=int, default=CFG.steps)
    ap.add_argument("--batch", type=int, default=CFG.batch)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--grad_accum", type=int, default=CFG.grad_accum)
    ap.add_argument("--log_every", type=int, default=CFG.log_every)

    # flat / pacf
    ap.add_argument("--flat_sigma", type=float, default=CFG.flat_sigma)
    ap.add_argument("--pac_lambda", type=float, default=CFG.pac_lambda)
    ap.add_argument("--reg_warmup_ratio", type=float, default=CFG.reg_warmup_ratio)
    ap.add_argument("--pacf_gamma", type=float, default=CFG.pacf_gamma)
    
    # eval
    ap.add_argument("--eval_steps", type=int, default=CFG.eval_steps)
    ap.add_argument("--eval_guidance", type=float, default=CFG.eval_guidance)
    ap.add_argument("--eval_prompts", type=str, nargs="*", default=list(CFG.eval_prompts))

    # infra
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--mixed_precision", type=str, default=CFG.mixed_precision, choices=["no", "fp16", "bf16"])
    ap.add_argument("--use_memory_efficient_attention", action="store_true", default=True)

    # W&B
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)

    # grid for Stage A
    ap.add_argument("--grid_lambda_points", type=str, default="")
    ap.add_argument("--grid_wr_points", type=str, default="")

    args = ap.parse_args()

    CFG.flat_sigma = args.flat_sigma
    CFG.image_size = args.image_size

    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    data_dir = Path(args.instance_data)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = ImageFolderDataset(data_dir, size=args.image_size)
    loader_full = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    set_all_seeds(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process and _WANDB and args.project:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    eval_prompts = [
        EvalPrompt(
            text=t,
            seed=args.seed + i,
            guidance=args.eval_guidance,
            steps=args.eval_steps,
        )
        for i, t in enumerate(args.eval_prompts)
    ]

    train_sched = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    # ===================== mode: single / seeds =====================
    if args.mode in ["single", "seeds"]:
        if args.mode == "single":
            seeds = [args.seed]
        else:
            if args.seeds:
                seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
            else:
                seeds = [args.seed]

        for s in seeds:
            pipe = build_pipe(
                args.pretrained,
                dtype,
                device,
                mem_eff=args.use_memory_efficient_attention,
            )
            add_lora_to_unet_and_textenc(
                pipe,
                rank=args.rank,
                alpha=args.alpha,
                lora_scope=args.lora_scope,
            )
            set_all_seeds(s)

            if args.variant == "lora":
                base_name = "lora"
                flat_sigma = 0.0
                use_pacf = False
                pac_lambda = 0.0
                reg_wr = 0.0
            elif args.variant == "flat":
                base_name = "flat"
                flat_sigma = args.flat_sigma
                use_pacf = False
                pac_lambda = 0.0
                reg_wr = 0.0
            else:
                base_name = "pacf"
                flat_sigma = 0.0
                use_pacf = True
                pac_lambda = args.pac_lambda
                reg_wr = args.reg_warmup_ratio

            stage_name = f"{args.mode}_{base_name}_seed{s}"
            print(f"\n== mode={args.mode}: {stage_name} ==")

            train_one_stage(
                accelerator=accelerator,
                stage_name=stage_name,
                pipe=pipe,
                dataloader=loader_full,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                flat_sigma=flat_sigma,
                flat_seed=s,
                use_pacf=use_pacf,
                pac_lambda=pac_lambda,
                reg_warmup_ratio=reg_wr,
                scheduler_train=train_sched,
                log_every=args.log_every,
                sample_prompts=eval_prompts,
                outdir=outdir,
                project=args.project,
                instance_prompt=args.instance_prompt,
                pacf_gamma=(args.pacf_gamma if use_pacf else 0.0),
            )

            grid_path = sample_and_grid(
                pipe,
                eval_prompts,
                outdir,
                f"{stage_name}_final",
            )
            score = try_clip_score(
                grid_path,
                [p.text for p in eval_prompts],
            )

            append_score_csv(outdir, {
                "mode": args.mode,
                "stage": "single",
                "variant": base_name,
                "seed": s,
                "flat_sigma": flat_sigma,
                "pac_lambda": pac_lambda,
                "reg_warmup_ratio": reg_wr,
                "clip_score": score,
            })

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if accelerator.is_main_process and _WANDB and args.project:
            wandb.finish()
        return

    # ===================== mode: auto (A/B/C pipeline) =====================
    plan = set([s.upper() for s in args.stage])
    best_pairs = None

    # ---------- Stage A ----------
    if "A" in plan:
        print("\n== Stage A: PACF grid on subset ==")

        k = max(1, int(len(dataset) * 0.35))
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        idxs = idxs[:k]
        subset = torch.utils.data.Subset(dataset, idxs)
        loader_sub = DataLoader(
            subset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        if args.grid_lambda_points.strip() and args.grid_wr_points.strip():
            gl = [float(x) for x in args.grid_lambda_points.split(",") if x.strip()]
            gw = [float(x) for x in args.grid_wr_points.split(",") if x.strip()]
            pairs = [(l, w) for l in gl for w in gw]
        else:
            pairs = [(5e-6, 0.25), (1e-5, 0.25), (2e-5, 0.30)]

        results = []
        cid = 0
        for lam, wr in pairs:
            cid += 1
            pipe = build_pipe(
                args.pretrained,
                dtype,
                device,
                mem_eff=args.use_memory_efficient_attention,
            )
            add_lora_to_unet_and_textenc(
                pipe,
                rank=args.rank,
                alpha=args.alpha,
                lora_scope=args.lora_scope,
            )
            set_all_seeds(args.seed)

            stage_name = f"A_c{cid}"
            train_one_stage(
                accelerator=accelerator,
                stage_name=stage_name,
                pipe=pipe,
                dataloader=loader_sub,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                flat_sigma=0.0,
                flat_seed=args.seed,
                use_pacf=True,
                pac_lambda=lam,
                reg_warmup_ratio=wr,
                scheduler_train=train_sched,
                log_every=args.log_every,
                sample_prompts=eval_prompts,
                outdir=outdir,
                project=args.project,
                instance_prompt=args.instance_prompt,
                pacf_gamma=args.pacf_gamma,
            )

            grid_path = sample_and_grid(
                pipe,
                eval_prompts,
                outdir,
                f"{stage_name}_final",
            )
            score = try_clip_score(
                grid_path,
                [p.text for p in eval_prompts],
            )
            print(f"[A] candidate {cid}: lam={lam:.1e} wr={wr:.2f} score={score:.4f}")

            append_score_csv(outdir, {
                "stage": "A",
                "candidate_id": cid,
                "variant": "pacf",
                "seed": args.seed,
                "pac_lambda": lam,
                "reg_warmup_ratio": wr,
                "flat_sigma": 0.0,
                "clip_score": score,
            })
            results.append({
                "cid": cid,
                "lam": lam,
                "wr": wr,
                "score": score,
            })

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        best_pairs = [(r["lam"], r["wr"]) for r in results_sorted[:max(1, args.topk_to_stageB)]]

        csv_path = outdir / "stageA_clip_scores.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cid", "lambda", "warmup", "clip_score"])
            for r in results_sorted:
                w.writerow([r["cid"], r["lam"], r["wr"], r["score"]])
        print(f"[A] scores saved to {csv_path}")

    # ---------- Stage B ----------
    best_lam, best_wr = (args.pac_lambda, args.reg_warmup_ratio)
    if "B" in plan:
        print("\n== Stage B: PACF best on full dataset ==")
        if not best_pairs:
            best_pairs = [(args.pac_lambda, args.reg_warmup_ratio)]
        lam, wr = best_pairs[0]
        best_lam, best_wr = lam, wr

        pipe = build_pipe(
            args.pretrained,
            dtype,
            device,
            mem_eff=args.use_memory_efficient_attention,
        )
        add_lora_to_unet_and_textenc(
            pipe,
            rank=args.rank,
            alpha=args.alpha,
            lora_scope=args.lora_scope,
        )
        set_all_seeds(args.seed)

        train_one_stage(
            accelerator=accelerator,
            stage_name="B_best",
            pipe=pipe,
            dataloader=loader_full,
            steps=args.steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
            flat_sigma=0.0,
            flat_seed=args.seed,
            use_pacf=True,
            pac_lambda=lam,
            reg_warmup_ratio=wr,
            scheduler_train=train_sched,
            log_every=args.log_every,
            sample_prompts=eval_prompts,
            outdir=outdir,
            project=args.project,
            instance_prompt=args.instance_prompt,
            pacf_gamma=args.pacf_gamma,

        )

        grid_path = sample_and_grid(
            pipe,
            eval_prompts,
            outdir,
            "B_final",
        )
        score = try_clip_score(
            grid_path,
            [p.text for p in eval_prompts],
        )
        append_score_csv(outdir, {
            "stage": "B",
            "variant": "pacf",
            "seed": args.seed,
            "pac_lambda": lam,
            "reg_warmup_ratio": wr,
            "flat_sigma": 0.0,
            "clip_score": score,
        })
        print(f"[B] best lam={lam:.1e}, wr={wr:.2f}, score={score:.4f}")

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- Stage C ----------
    if "C" in plan:
        print("\n== Stage C: multi-seed LoRA / Flat / PACF ==")
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

        # LoRA baseline
        for s in seeds:
            pipe = build_pipe(
                args.pretrained,
                dtype,
                device,
                mem_eff=args.use_memory_efficient_attention,
            )
            add_lora_to_unet_and_textenc(
                pipe,
                rank=args.rank,
                alpha=args.alpha,
                lora_scope=args.lora_scope,
            )
            set_all_seeds(s)

            stage_name = f"C_LoRA_seed{s}"
            train_one_stage(
                accelerator=accelerator,
                stage_name=stage_name,
                pipe=pipe,
                dataloader=loader_full,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                flat_sigma=0.0,
                flat_seed=s,
                use_pacf=False,
                pac_lambda=0.0,
                reg_warmup_ratio=0.0,
                scheduler_train=train_sched,
                log_every=args.log_every,
                sample_prompts=eval_prompts,
                outdir=outdir,
                project=args.project,
                instance_prompt=args.instance_prompt,
                pacf_gamma=0.0,
            )

            grid_path = sample_and_grid(
                pipe,
                eval_prompts,
                outdir,
                f"{stage_name}_final",
            )
            score = try_clip_score(
                grid_path,
                [p.text for p in eval_prompts],
            )
            append_score_csv(outdir, {
                "stage": "C",
                "variant": "lora",
                "seed": s,
                "pac_lambda": 0.0,
                "reg_warmup_ratio": 0.0,
                "flat_sigma": 0.0,
                "clip_score": score,
            })

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Flat-LoRA baseline
        for s in seeds:
            pipe = build_pipe(
                args.pretrained,
                dtype,
                device,
                mem_eff=args.use_memory_efficient_attention,
            )
            add_lora_to_unet_and_textenc(
                pipe,
                rank=args.rank,
                alpha=args.alpha,
                lora_scope=args.lora_scope,
            )
            set_all_seeds(s)

            stage_name = f"C_Flat_seed{s}"
            train_one_stage(
                accelerator=accelerator,
                stage_name=stage_name,
                pipe=pipe,
                dataloader=loader_full,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                flat_sigma=CFG.flat_sigma,
                flat_seed=s,
                use_pacf=False,
                pac_lambda=0.0,
                reg_warmup_ratio=0.0,
                scheduler_train=train_sched,
                log_every=args.log_every,
                sample_prompts=eval_prompts,
                pacf_gamma=0.0,
                outdir=outdir,
                project=args.project,
                instance_prompt=args.instance_prompt,
            )

            grid_path = sample_and_grid(
                pipe,
                eval_prompts,
                outdir,
                f"{stage_name}_final",
            )
            score = try_clip_score(
                grid_path,
                [p.text for p in eval_prompts],
            )
            append_score_csv(outdir, {
                "stage": "C",
                "variant": "flat",
                "seed": s,
                "pac_lambda": 0.0,
                "reg_warmup_ratio": 0.0,
                "flat_sigma": CFG.flat_sigma,
                "clip_score": score,
            })

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # PACF (best lam, wr from B/A)
        for s in seeds:
            pipe = build_pipe(
                args.pretrained,
                dtype,
                device,
                mem_eff=args.use_memory_efficient_attention,
            )
            add_lora_to_unet_and_textenc(
                pipe,
                rank=args.rank,
                alpha=args.alpha,
                lora_scope=args.lora_scope,
            )
            set_all_seeds(s)

            stage_name = f"C_PACF_seed{s}"
            train_one_stage(
                accelerator=accelerator,
                stage_name=stage_name,
                pipe=pipe,
                dataloader=loader_full,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                flat_sigma=0.0,
                flat_seed=s,
                use_pacf=True,
                pac_lambda=best_lam,
                reg_warmup_ratio=best_wr,
                scheduler_train=train_sched,
                log_every=args.log_every,
                sample_prompts=eval_prompts,
                outdir=outdir,
                project=args.project,
                instance_prompt=args.instance_prompt,
                pacf_gamma=args.pacf_gamma,
            )

            grid_path = sample_and_grid(
                pipe,
                eval_prompts,
                outdir,
                f"{stage_name}_final",
            )
            score = try_clip_score(
                grid_path,
                [p.text for p in eval_prompts],
            )
            append_score_csv(outdir, {
                "stage": "C",
                "variant": "pacf",
                "seed": s,
                "pac_lambda": best_lam,
                "reg_warmup_ratio": best_wr,
                "flat_sigma": 0.0,
                "clip_score": score,
            })

            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process and _WANDB and args.project:
        wandb.finish()


if __name__ == "__main__":
    main()