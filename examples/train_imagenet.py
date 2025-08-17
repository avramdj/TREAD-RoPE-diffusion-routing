from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path
import json
import numpy as np

from diffusers import AutoencoderKL
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb
from tread_diffusion import DiT, DiT_models

# Imagenet.int8: Entire Imagenet dataset in 5GB
# See: https://github.com/SonicCodes/imagenet.int8 (shoutout to rami and simo)
class ImageNetInt8LatentDataset(torch.utils.data.Dataset):
    """
    Memory-mapped ImageNet latents compressed to uint8 using SDXL VAE.
    Each sample is 4096 bytes (4x32x32) stored as uint8.
    Latents are dequantized to float32 using (x/255 - 0.5) * 24.0
    and returned as shape [4, 32, 32].
    Labels file is a JSON list of [label_index, label_text].
    """

    def __init__(self, data_path: str, labels_path: str):
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        num_samples = len(self.labels)
        self.data = np.memmap(
            data_path, dtype="uint8", mode="r", shape=(num_samples, 4096)
        )

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
        self.last_fid_images = 0
        self.last_imglog_images = 0
        self.last_ckpt_images = 0

    def log_step(self, *, loss_value: float, lr: float, step: int, images_seen: int) -> None:
        wandb.log({
            "train/loss": loss_value,
            "lr": lr,
            "step": step,
            "images_seen": images_seen,
        })
        if (images_seen - self.last_fid_images) >= self.log_fid_every_images:
            self._compute_and_log_fid(step=step, images_seen=images_seen)
            self.last_fid_images = images_seen
        if (images_seen - self.last_imglog_images) >= self.log_images_every_images:
            self._log_sample_images(step=step, images_seen=images_seen)
            self.last_imglog_images = images_seen
        if (images_seen - self.last_ckpt_images) >= self.save_every_images:
            self._save_checkpoint(step=step, images_seen=images_seen)
            self.last_ckpt_images = images_seen

    @staticmethod
    def _fmt_k(value: float) -> str:
        if value >= 1000:
            return f"{value/1000.0:.2f}K"
        return f"{value:.0f}"

    def _images_to_steps(self, images: int) -> int:
        return int(math.ceil(images / float(max(1, self.batch_size))))

    def format_status(self, images_seen: int) -> str:
        seen_str = self._fmt_k(float(images_seen))
        fid_rem_imgs = max(0, (self.last_fid_images + self.log_fid_every_images) - images_seen)
        img_rem_imgs = max(0, (self.last_imglog_images + self.log_images_every_images) - images_seen)
        ckp_rem_imgs = max(0, (self.last_ckpt_images + self.save_every_images) - images_seen)
        fid_steps = self._images_to_steps(fid_rem_imgs)
        img_steps = self._images_to_steps(img_rem_imgs)
        ckp_steps = self._images_to_steps(ckp_rem_imgs)
        return f"seen {seen_str} | F {self._fmt_k(fid_steps)} | I {self._fmt_k(img_steps)} | C {self._fmt_k(ckp_steps)}"

    @torch.no_grad()
    def _compute_and_log_fid(self, *, step: int, images_seen: int) -> None:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        real_added = 0
        for latents_eval, _ in self.train_loader:
            latents_eval = latents_eval.to(self.device)
            decoded = self.vae.decode(latents_eval).sample  # [-1,1]
            real = (decoded + 1.0) * 0.5
            real = torch.nn.functional.interpolate(
                real, size=(299, 299), mode="bilinear", align_corners=False
            )
            fid.update(real.clamp(0, 1), real=True)
            real_added += real.shape[0]
            if real_added >= self.fid_samples:
                break

        n = min(self.fid_samples, self.train_loader.batch_size or self.fid_samples)
        remain = self.fid_samples
        while remain > 0:
            b = min(remain, n)
            y = torch.randint(0, 1000, (b,), device=self.device)
            x_t = self.ema_model.sample_ode_rectified_euler(
                b, height=32, width=32, num_steps=self.timesteps, class_labels=y
            )
            decoded_fake = self.vae.decode(x_t).sample
            fake_01 = (decoded_fake + 1.0) * 0.5
            fake_for_fid = torch.nn.functional.interpolate(
                fake_01, size=(299, 299), mode="bilinear", align_corners=False
            )
            fid.update(fake_for_fid.clamp(0, 1), real=False)
            remain -= b

        fid_value = float(fid.compute().item())
        wandb.log({"eval/fid": fid_value, "step": step, "images_seen": images_seen})

    @torch.no_grad()
    def _log_sample_images(self, *, step: int, images_seen: int) -> None:
        b = self.grid_n
        y = torch.randint(0, 1000, (b,), device=self.device)
        x_t = self.model.sample_ode_rectified_euler(
            num_images=b, height=32, width=32, num_steps=self.timesteps, class_labels=y
        )
        decoded = self.vae.decode(x_t).sample
        imgs = (decoded + 1.0) * 0.5
        grid = make_grid(
            imgs.detach().cpu()[: self.grid_n],
            nrow=int(math.sqrt(self.grid_n)),
            padding=2,
            normalize=True,
            value_range=(0, 1),
        )
        wandb.log({"viz/samples_grid": wandb.Image(grid), "step": step, "images_seen": images_seen})

        x_t_diffeq = self.model.sample_ode_rectified_diffeq(
            num_images=b, height=32, width=32, num_steps=self.timesteps, class_labels=y
        )
        decoded_diffeq = self.vae.decode(x_t_diffeq).sample
        imgs_diffeq = (decoded_diffeq + 1.0) * 0.5
        grid_diffeq = make_grid(
            imgs_diffeq.detach().cpu()[: self.grid_n],
            nrow=int(math.sqrt(self.grid_n)),
            padding=2,
            normalize=True,
            value_range=(0, 1),
        )
        wandb.log({"viz/samples_grid_diffeq": wandb.Image(grid_diffeq), "step": step, "images_seen": images_seen})

    def _save_checkpoint(self, *, step: int, images_seen: int) -> None:
        ckpt_path = self.save_dir / f"dit_imagenet_step{step}.pt"
        torch.save({"model": self.model.state_dict(), "ema": self.ema_model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    return betas


def precompute_alphas(betas: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, alphas


@torch.no_grad()
def q_sample(x0: Tensor, t: Tensor, noise: Tensor, sqrt_ac: Tensor, sqrt_1m_ac: Tensor) -> Tensor:
    s1 = sqrt_ac.gather(0, t).view(-1, 1, 1, 1)
    s2 = sqrt_1m_ac.gather(0, t).view(-1, 1, 1, 1)
    return s1 * x0 + s2 * noise


@torch.no_grad()
def sample_model(
    model: DiT,
    *,
    num_images: int,
    num_steps: int,
    classes: Tensor | None = None,
    height: int = 32,
    width: int = 32,
) -> Tensor:
    return model.sample_ode_rectified_euler(
        num_images,
        height=height,
        width=width,
        num_steps=num_steps,
        class_labels=classes.long() if classes is not None else None,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train DiT on ImageNet.int8 latents (SDXL VAE)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--rope", type=str, default="none", choices=["none", "axial", "golden_gate"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="examples/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-samples", type=int, default=256)
    # Step/image-based intervals (defaults scaled by batch size after parsing)
    parser.add_argument("--log-fid-every", type=int, default=None)
    parser.add_argument("--log-images-every", type=int, default=None)
    parser.add_argument("--grid-n", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--dit-size", type=str, default="DiT-S/2", choices=DiT_models.keys())
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Scale default intervals by batch size if not provided
    if args.log_fid_every is None:
        args.log_fid_every = 10000 * args.batch_size
    if args.log_images_every is None:
        args.log_images_every = 1000 * args.batch_size
    if args.save_every is None:
        args.save_every = 10000 * args.batch_size

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
    route_config = {"start_block": 2, "end_block": 10, "rate": 0.5, "mix_factor": 0.5}
    model = DiT_models[args.dit_size](
        input_size=32,
        in_channels=4,  # SDXL latent channels
        num_classes=1000,
        learn_sigma=False,
        rope=rope_arg,
        route_config=route_config,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # EMA model (kept in eval mode)
    ema_model = DiT_models[args.dit_size](
        input_size=32,
        in_channels=4,
        num_classes=1000,
        learn_sigma=False,
        rope=rope_arg,
        route_config=route_config,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(ema: DiT, online: DiT, decay: float) -> None:
        for ema_p, p in zip(ema.parameters(), online.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        # Copy buffers (for completeness)
        for ema_buf, buf in zip(ema.buffers(), online.buffers()):
            ema_buf.copy_(buf)
    

    # Compile and precision stuff
    from torch._inductor import config as inductor_config

    torch.set_float32_matmul_precision("high")
    inductor_config.triton.cudagraphs = False
    model = torch.compile(model, dynamic=True, fullgraph=False, mode="max-autotune-no-cudagraphs")
    
    # RECTIFIED FLOW

    wandb_project = os.getenv("WANDB_PROJECT", "tread-diffusion-imagenet-int8")
    wandb.init(
        project=wandb_project,
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

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )

    # LR warmup (5%) + linear decay to 0 by end
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        step = step + 1
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    global_step = 0
    images_seen = 0

    # Reuse a single VAE instance for decoding
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()

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
        log_fid_every_images=args.log_fid_every,
        log_images_every_images=args.log_images_every,
        save_every_images=args.save_every,
        batch_size=args.batch_size,
    )

    # Print concise hook schedule (in steps)
    def fmt_k(val: int) -> str:
        return f"{val/1000.0:.2f}K" if val >= 1000 else f"{val}"
    fid_steps = int(math.ceil(args.log_fid_every / float(max(1, args.batch_size))))
    img_steps = int(math.ceil(args.log_images_every / float(max(1, args.batch_size))))
    ckp_steps = int(math.ceil(args.save_every / float(max(1, args.batch_size))))
    print(
        f"Logging: FID every {fmt_k(fid_steps)} steps, "
        f"Images every {fmt_k(img_steps)} steps, "
        f"Ckpt every {fmt_k(ckp_steps)} steps. Batch={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=logger.format_status(images_seen), leave=False)
        for latents, labels in pbar:
            latents = latents.to(device)
            labels = labels.to(device)

            # Rectified flow in latent space
            x1 = torch.randn_like(latents)
            t = torch.rand(latents.shape[0], device=device)
            x_t = (1.0 - t).view(-1, 1, 1, 1) * x1 + t.view(-1, 1, 1, 1) * latents
            v_target = latents - x1

            with amp_ctx:
                pred_v = model(x_t, t, labels)
                loss = F.mse_loss(pred_v, v_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_update(ema_model, model, args.ema_decay)
            scheduler.step()
            global_step += 1
            images_seen += latents.shape[0]

            running_loss += loss.item()
            num_batches += 1
            tqdm.write("") if False else None

            # Unified logging/triggering
            logger.log_step(
                loss_value=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                step=global_step,
                images_seen=images_seen,
            )

            # Update concise tqdm status
            pbar.set_description(logger.format_status(images_seen))


if __name__ == "__main__":
    main()
