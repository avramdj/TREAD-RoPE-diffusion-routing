from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from dotenv import load_dotenv
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb
from tread_diffusion import DiT, DiT_models
from tread_diffusion.rectified_flow import RectifiedFlow


# Imagenet.int8: Entire Imagenet dataset in 5GB
# See: https://github.com/SonicCodes/imagenet.int8 (shoutout to rami and simo)
class ImageNetInt8LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, labels_path: str):
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        num_samples = len(self.labels)
        self.data = np.memmap(data_path, dtype="uint8", mode="r", shape=(num_samples, 4096))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        vec = self.data[idx]
        x = vec.astype(np.float32).reshape(4, 32, 32)
        x = (x / 255.0 - 0.5) * 24.0
        label_idx = int(self.labels[idx][0])
        return torch.from_numpy(x), label_idx


class TrainingLogger:
    """Encapsulates all training-time logging and checkpointing based on image counts."""

    def __init__(
        self,
        *,
        device: torch.device,
        vae: AutoencoderKL,
        train_loader: DataLoader,
        model: DiT,
        ema_model: DiT,
        grid_n: int,
        timesteps: int,
        fid_samples: int,
        save_dir: Path,
        log_fid_every_images: int,
        log_images_every_images: int,
        save_every_images: int,
        batch_size: int,
        sample_seed: int,
        rf: RectifiedFlow,
        save_name: str,
    ) -> None:
        self.device = device
        self.vae = vae
        self.train_loader = train_loader
        self.model = model
        self.ema_model = ema_model
        self.grid_n = grid_n
        self.timesteps = timesteps
        self.fid_samples = fid_samples
        self.save_dir = save_dir
        self.log_fid_every_images = log_fid_every_images
        self.log_images_every_images = log_images_every_images
        self.save_every_images = save_every_images
        self.batch_size = batch_size
        self.sample_seed = sample_seed
        self.last_fid_images = 0
        self.last_imglog_images = 0
        self.last_ckpt_images = 0
        self.real_logged_once = False
        self.rf = rf
        self.save_name = save_name

    def log_step(
        self,
        *,
        loss_value: float,
        lr: float,
        step: int,
        images_seen: int,
        grad_norm: float,
        post_clip_norm: float,
    ) -> None:
        wandb.log(
            {
                "train/loss": loss_value,
                "lr": lr,
                "step": step,
                "images_seen": images_seen,
                "grad_norm": grad_norm,
                "post_clip_norm": post_clip_norm,
            }
        )
        if (
            images_seen - self.last_fid_images
        ) >= self.log_fid_every_images and self.log_fid_every_images != -1:
            self._compute_and_log_fid(step=step, images_seen=images_seen)
            self.last_fid_images = images_seen
        if (
            images_seen - self.last_imglog_images
        ) >= self.log_images_every_images and self.log_images_every_images != -1:
            self._log_sample_images(step=step, images_seen=images_seen)
            self.last_imglog_images = images_seen
        if (
            images_seen - self.last_ckpt_images
        ) >= self.save_every_images and self.save_every_images != -1:
            self._save_checkpoint(step=step, images_seen=images_seen)
            self.last_ckpt_images = images_seen

    @staticmethod
    def _fmt_k(value: float) -> str:
        if value >= 1000:
            return f"{value / 1000.0:.2f}K"
        return f"{value:.0f}"

    def _images_to_steps(self, images: int) -> int:
        return int(math.ceil(images / float(max(1, self.batch_size))))

    def format_status(self, images_seen: int) -> str:
        seen_str = self._fmt_k(float(images_seen))
        fid_rem_imgs = max(0, (self.last_fid_images + self.log_fid_every_images) - images_seen)
        img_rem_imgs = max(0, (self.last_imglog_images + self.log_images_every_images) - images_seen)
        ckp_rem_imgs = max(0, (self.last_ckpt_images + self.save_every_images) - images_seen)
        fid_steps = self._images_to_steps(fid_rem_imgs) if self.log_fid_every_images != -1 else -1
        img_steps = self._images_to_steps(img_rem_imgs) if self.log_images_every_images != -1 else -1
        ckp_steps = self._images_to_steps(ckp_rem_imgs) if self.save_every_images != -1 else -1
        ret = f"seen {seen_str}"
        if self.log_fid_every_images != -1:
            ret += f" | F {self._fmt_k(fid_steps)}"
        if self.log_images_every_images != -1:
            ret += f" | I {self._fmt_k(img_steps)}"
        if self.save_every_images != -1:
            ret += f" | C {self._fmt_k(ckp_steps)}"
        return ret

    @torch.no_grad()
    def _compute_and_log_fid(self, *, step: int, images_seen: int) -> None:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        real_added = 0
        for latents_eval, _ in self.train_loader:
            latents_eval = latents_eval.to(self.device)
            real = torch.nn.functional.interpolate(
                decode_latents(latents_eval, self.vae),
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )
            fid.update(real, real=True)
            real_added += real.shape[0]
            if real_added >= self.fid_samples:
                break

        n = min(self.fid_samples, self.train_loader.batch_size or self.fid_samples)
        remain = self.fid_samples
        while remain > 0:
            b = min(remain, n)
            y = torch.randint(0, 1000, (b,), device=self.device)
            x_t = self.rf.sample_euler(num_images=b, model=self.ema_model, class_labels=y)
            fake_for_fid = torch.nn.functional.interpolate(
                decode_latents(x_t, self.vae), size=(299, 299), mode="bilinear", align_corners=False
            )
            fid.update(fake_for_fid, real=False)
            remain -= b

        fid_value = float(fid.compute().item())
        wandb.log({"eval/fid": fid_value, "step": step, "images_seen": images_seen})

    @torch.no_grad()
    def _log_sample_images(self, *, step: int, images_seen: int) -> None:
        b = self.grid_n
        # Save and restore RNG state to keep sampling deterministic without affecting training RNG
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        torch.manual_seed(self.sample_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.sample_seed)

        def log_grid(tag: str, imgs: Tensor) -> None:
            grid = make_grid(
                imgs.detach().cpu()[: self.grid_n],
                nrow=int(math.sqrt(self.grid_n)),
                padding=2,
                normalize=True,
                value_range=(0, 1),
            )
            wandb.log({tag: wandb.Image(grid), "step": step, "images_seen": images_seen})

        y = torch.randint(0, 1000, (b,), device=self.device)
        imgs = decode_latents(
            self.rf.sample_euler(num_images=b, model=self.model, class_labels=y), self.vae
        )
        log_grid("viz/euler", imgs)
        imgs_diffeq = decode_latents(
            self.rf.sample_diffeq(num_images=b, model=self.model, class_labels=y), self.vae
        )
        log_grid("viz/diffeq", imgs_diffeq)
        imgs_cfg = decode_latents(
            self.rf.sample_euler(num_images=b, model=self.model, class_labels=y, cfg_scale=3.0),
            self.vae,
        )
        log_grid("viz/diffeq_cfg_3", imgs_cfg)
        imgs_cfg = decode_latents(
            self.rf.sample_diffeq(
                num_images=b, model=self.model, class_labels=y, cfg_scale=3.0, apg=True
            ),
            self.vae,
        )
        log_grid("viz/diffeq_cfg_3_apg", imgs_cfg)
        imgs_ema = decode_latents(
            self.rf.sample_euler(num_images=b, model=self.ema_model, class_labels=y), self.vae
        )
        log_grid("viz/ema", imgs_ema)
        # Also log a grid of ground-truth images decoded from real latents
        if not self.real_logged_once:
            try:
                latents_real, _ = next(iter(self.train_loader))
                latents_real = latents_real.to(self.device)
                imgs_real = decode_latents(latents_real[:b], self.vae)
                grid_real = make_grid(
                    imgs_real.detach().cpu()[: self.grid_n],
                    nrow=int(math.sqrt(self.grid_n)),
                    padding=2,
                    normalize=True,
                    value_range=(0, 1),
                )
                wandb.log(
                    {
                        "viz/real": wandb.Image(grid_real),
                        "step": step,
                        "images_seen": images_seen,
                    }
                )
                self.real_logged_once = True
            except StopIteration:
                pass
        # Restore RNG states
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, self.device)

    def _save_checkpoint(self, *, step: int, images_seen: int) -> None:
        ckpt_path = self.save_dir / f"{self.save_name}.pt"
        torch.save({"model": self.model.state_dict(), "ema": self.ema_model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


def decode_latents(latents: Tensor, vae: AutoencoderKL) -> Tensor:
    # return ((vae.decode((latents).to(vae.dtype)).sample + 1.0) * 0.5).clamp(0, 1)
    vae_sample = vae.decode((latents).to(vae.dtype)).sample
    return VaeImageProcessor().postprocess(
        image=vae_sample,
        do_denormalize=[True] * vae_sample.shape[0],
        output_type="pt",
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train DiT on ImageNet.int8 latents (SDXL VAE)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--rope", type=str, default="none", choices=["none", "axial", "golden_gate"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="examples/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-samples", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    # Interval configuration
    parser.add_argument(
        "--log-fid-every", type=int, default=None, help="Interval value; unit set by --interval-unit"
    )
    parser.add_argument(
        "--log-images-every", type=int, default=None, help="Interval value; unit set by --interval-unit"
    )
    parser.add_argument("--grid-n", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument(
        "--save-every", type=int, default=None, help="Interval value; unit set by --interval-unit"
    )
    parser.add_argument("--interval-unit", type=str, choices=["steps", "images"], default="steps")
    parser.add_argument("--dit-size", type=str, default="DiT-S/2", choices=DiT_models.keys())
    parser.add_argument("--start-block", type=int, default=2)
    parser.add_argument("--end-block", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    # ImageNet.int8 dataset paths
    parser.add_argument(
        "--inet-data",
        type=str,
        default="examples/data/imagenet_int8/inet.npy",
        help="Path to ImageNet.int8 memmap .npy (uint8) file",
    )
    parser.add_argument(
        "--inet-labels",
        type=str,
        default="examples/data/imagenet_int8/inet.json",
        help="Path to ImageNet.int8 labels .json file",
    )
    # SDXL VAE for decoding latents to pixels for FID/logging
    parser.add_argument(
        "--vae",
        type=str,
        default="stabilityai/sdxl-vae",
        help="Hugging Face id or local path for SDXL VAE",
    )
    parser.add_argument("--wandb-name", type=str, default="tread-diffusion-imagenet-int8")
    parser.add_argument("--save-name", type=str, default="dit-latest")
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint .pt to load (optional)"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Resolve intervals: defaults in steps, convert to images if needed
    default_fid_steps = 10000
    default_img_steps = 1000
    default_ckpt_steps = 10000
    fid_steps = args.log_fid_every if args.log_fid_every is not None else default_fid_steps
    img_steps = args.log_images_every if args.log_images_every is not None else default_img_steps
    ckp_steps = args.save_every if args.save_every is not None else default_ckpt_steps
    if args.interval_unit == "steps":
        log_fid_every_images = fid_steps * args.batch_size if fid_steps != -1 else -1
        log_images_every_images = img_steps * args.batch_size if img_steps != -1 else -1
        save_every_images = ckp_steps * args.batch_size if ckp_steps != -1 else -1
    else:
        # provided values are in images; also compute step views for display
        log_fid_every_images = fid_steps
        log_images_every_images = img_steps
        save_every_images = ckp_steps
        fid_steps = (
            int(math.ceil(fid_steps / float(max(1, args.batch_size)))) if fid_steps != -1 else -1
        )
        img_steps = (
            int(math.ceil(img_steps / float(max(1, args.batch_size)))) if img_steps != -1 else -1
        )
        ckp_steps = (
            int(math.ceil(ckp_steps / float(max(1, args.batch_size)))) if ckp_steps != -1 else -1
        )

    # ImageNet.int8 latent dataset
    train_ds = ImageNetInt8LatentDataset(args.inet_data, args.inet_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Small DiT (reasonable defaults)
    rope_arg = None if args.rope == "none" else args.rope
    route_config = {
        "start_block": args.start_block,
        "end_block": args.end_block,
        "rate": 0.5,
        "mix_factor": 0.5,
    }
    dit_args = {
        "input_size": 32,
        "in_channels": 4,  # SDXL latent channels
        "num_classes": 1000,
        "learn_sigma": False,
        "rope": rope_arg,
        "route_config": route_config,
    }
    model = DiT_models[args.dit_size](**dit_args).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.001)

    # EMA model (kept in eval mode)
    ema_model = DiT_models[args.dit_size](**dit_args).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Optionally load checkpoint BEFORE compile to avoid _orig_mod key mismatch
    if args.ckpt is not None and len(args.ckpt) > 0:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model_state = ckpt.get("model", ckpt)
        try:
            model.load_state_dict(model_state, strict=True)
        except Exception:
            stripped = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
            model.load_state_dict(stripped, strict=True)
        ema_state = ckpt.get("ema")
        if ema_state is not None:
            try:
                ema_model.load_state_dict(ema_state, strict=True)
            except Exception:
                stripped_ema = {k.replace("_orig_mod.", ""): v for k, v in ema_state.items()}
                ema_model.load_state_dict(stripped_ema, strict=True)
        else:
            ema_model.load_state_dict(model.state_dict())
        print(f"Loaded checkpoint from {args.ckpt}")

    @torch.no_grad()
    def ema_update(ema: DiT, online: DiT, decay: float, compiled: bool = True) -> None:
        ema_params = OrderedDict(ema.named_parameters())
        online_params = OrderedDict(online.named_parameters())
        for name, param in online_params.items():
            if param.requires_grad:
                ema_name = name.replace("_orig_mod.", "") if compiled else name
                ema_params[ema_name].mul_(decay).add_(online_params[name].data, alpha=1 - decay)

    # Compile and precision stuff
    from torch._inductor import config as inductor_config

    torch.set_float32_matmul_precision("high")
    inductor_config.triton.cudagraphs = False
    model = torch.compile(model, dynamic=False, fullgraph=False, mode="max-autotune")

    # RECTIFIED FLOW

    wandb_project = os.getenv("WANDB_PROJECT", "tread-diffusion-imagenet-int8")
    wandb.init(
        project=wandb_project,
        name=args.wandb_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timesteps": args.timesteps,
            "rope": args.rope,
            "route_config": route_config,
            "inet_data": args.inet_data,
            "inet_labels": args.inet_labels,
            "vae": args.vae,
        },
    )

    model.train()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        assert torch.cuda.is_bf16_supported(), "bf16 is not supported on this GPU broke boy"
    use_amp = device.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )

    # lr scheduler (in optimizer steps, respects gradient accumulation)
    accum_steps = max(1, args.grad_accum)
    total_steps = max(1, int(math.ceil(args.epochs * len(train_loader) / float(accum_steps))))
    warmup_steps = max(1, int(math.ceil(len(train_loader) / float(accum_steps))))

    def lr_lambda(step: int) -> float:
        step = step + 1
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    global_step = 0  # counts optimizer steps
    images_seen = 0

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()

    rf = RectifiedFlow(
        height=32,
        width=32,
        num_steps=args.timesteps,
        vae_scaling_factor=vae.config.scaling_factor,
    )

    logger = TrainingLogger(
        device=device,
        vae=vae,
        train_loader=train_loader,
        model=model,
        ema_model=ema_model,
        grid_n=args.grid_n,
        timesteps=args.timesteps,
        fid_samples=args.fid_samples,
        save_dir=Path(args.save_dir),
        log_fid_every_images=log_fid_every_images,
        log_images_every_images=log_images_every_images,
        save_every_images=save_every_images,
        batch_size=args.batch_size,
        sample_seed=1337,
        rf=rf,
        save_name=args.save_name,
    )

    # Print concise hook schedule (in steps)
    def fmt_k(val: int) -> str:
        return f"{val / 1000.0:.2f}K" if val >= 1000 else f"{val}"

    print(
        f"Logging: FID every {fmt_k(fid_steps)} steps, "
        f"Images every {fmt_k(img_steps)} steps, "
        f"Ckpt every {fmt_k(ckp_steps)} steps. Batch={args.batch_size}"
    )

    total_steps = args.epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc=logger.format_status(images_seen), leave=True)
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        num_batches = 0
        accum_loss = 0.0
        accum_count = 0
        optimizer.zero_grad()
        for latents, labels in train_loader:
            latents = latents.to(device)
            labels = labels.to(device)

            loss = rf.loss(model, latents, labels, amp_ctx=amp_ctx)

            (loss / float(accum_steps)).backward()
            images_seen += latents.shape[0]

            running_loss += loss.item()
            accum_loss += loss.item()
            accum_count += 1
            num_batches += 1
            tqdm.write("") if False else None

            step_now = (num_batches % accum_steps) == 0
            if step_now:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_grad_norm
                )
                post_clip_norm = torch.norm(
                    torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), 2
                )
                optimizer.step()
                ema_update(ema_model, model, args.ema_decay)
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Unified logging/triggering (average loss over micro-batches)
                avg_loss = accum_loss / float(max(1, accum_count))
                logger.log_step(
                    loss_value=avg_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    step=global_step,
                    images_seen=images_seen,
                    grad_norm=grad_norm.item(),
                    post_clip_norm=post_clip_norm.item(),
                )
                accum_loss = 0.0
                accum_count = 0

            # Update concise tqdm status
            pbar.set_description(logger.format_status(images_seen))
            pbar.update(1)

        # Flush leftover grads at epoch end
        if (num_batches % accum_steps) != 0 and accum_count > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            post_clip_norm = torch.norm(
                torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), 2
            )
            optimizer.step()
            ema_update(ema_model, model, args.ema_decay)
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_loss = accum_loss / float(max(1, accum_count))
            logger.log_step(
                loss_value=avg_loss,
                lr=optimizer.param_groups[0]["lr"],
                step=global_step,
                images_seen=images_seen,
                grad_norm=grad_norm.item(),
                post_clip_norm=post_clip_norm.item(),
            )
    pbar.close()


if __name__ == "__main__":
    main()
