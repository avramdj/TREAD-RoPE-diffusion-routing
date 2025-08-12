from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This example requires torchvision. Install with `uv add torchvision` or `pip install torchvision`."
    ) from e

import wandb
from dotenv import load_dotenv
from torchmetrics.image.fid import FrechetInceptionDistance

from tread_diffusion import DiT


def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    return betas


def precompute_alphas(betas: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, alphas


@torch.no_grad()
def q_sample(x0: Tensor, t: Tensor, noise: Tensor, sqrt_ac: Tensor, sqrt_1m_ac: Tensor) -> Tensor:
    # Gather per batch
    s1 = sqrt_ac.gather(0, t).view(-1, 1, 1, 1)
    s2 = sqrt_1m_ac.gather(0, t).view(-1, 1, 1, 1)
    return s1 * x0 + s2 * noise


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train DiT on MNIST (pixel space)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--rope", type=str, default="none", choices=["none", "axial", "golden_gate"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="examples/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-samples", type=int, default=256)
    parser.add_argument("--log-fid-every", type=int, default=1)
    parser.add_argument("--log-images-every", type=int, default=5)
    parser.add_argument("--grid-n", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Data
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),  # scale to [-1, 1]
        ]
    )
    train_ds = datasets.MNIST(root="examples/data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    rope_arg = None if args.rope == "none" else args.rope
    model = DiT(
        input_size=28,
        patch_size=2,
        in_channels=1,
        hidden_size=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
        learn_sigma=False,
        rope=rope_arg,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    betas = make_beta_schedule(args.timesteps).to(device)
    sqrt_ac, sqrt_1m_ac, _ = precompute_alphas(betas)

    wandb_project = os.getenv("WANDB_PROJECT", "tread-diffusion-mnist")
    wandb.init(
        project=wandb_project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timesteps": args.timesteps,
            "rope": args.rope,
            "model_hidden": 192,
            "depth": 6,
            "num_heads": 6,
        },
    )

    model.train()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images = images.to(device)  # [B,1,28,28]
            labels = labels.to(device)

            # Sample random timesteps per batch element
            t = torch.randint(0, args.timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            x_t = q_sample(images, t, noise, sqrt_ac, sqrt_1m_ac)

            # Forward (DiT expects t as float tensor)
            t_float = t.float()
            pred_noise = model(x_t, t_float, labels)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg = running_loss / max(1, num_batches)
        wandb.log({"train/loss": avg, "epoch": epoch})
        print(f"Epoch {epoch}: loss={avg:.4f}")

        # FID evaluation
        if epoch % args.log_fid_every == 0:
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            model.eval()
            # Real images
            real_added = 0
            with torch.no_grad():
                for images, _ in train_loader:
                    images = images.to(device)
                    real = (images + 1.0) * 0.5
                    real3 = real.repeat(1, 3, 1, 1)
                    real3 = torch.nn.functional.interpolate(
                        real3, size=(299, 299), mode="bilinear", align_corners=False
                    )
                    fid.update(real3.clamp(0, 1), real=True)
                    real_added += real3.shape[0]
                    if real_added >= args.fid_samples:
                        break

                # Fake images via simple DDPM sampling
                n = min(args.fid_samples, args.batch_size)
                remain = args.fid_samples
                sample_images = []
                while remain > 0:
                    b = min(remain, n)
                    x_t = torch.randn(b, 1, 28, 28, device=device)
                    y = torch.randint(0, 10, (b,), device=device)
                    for t_step in reversed(range(args.timesteps)):
                        t = torch.full((b,), float(t_step), device=device)
                        eps = model(x_t, t, y)
                        beta_t = betas[t_step]
                        alpha_t = 1.0 - beta_t
                        alpha_cum_t = (1.0 - betas[: t_step + 1]).prod()
                        mean = (1.0 / math.sqrt(alpha_t)) * (
                            x_t - (beta_t / math.sqrt(1 - alpha_cum_t)) * eps
                        )
                        if t_step > 0:
                            x_t = mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
                        else:
                            x_t = mean
                    fake = (x_t + 1.0) * 0.5
                    if epoch % args.log_images_every == 0 and len(sample_images) < args.grid_n:
                        take = min(args.grid_n - len(sample_images), fake.shape[0])
                        sample_images.append(fake[:take].detach().cpu())
                    fake3 = fake.repeat(1, 3, 1, 1)
                    fake3 = torch.nn.functional.interpolate(
                        fake3, size=(299, 299), mode="bilinear", align_corners=False
                    )
                    fid.update(fake3.clamp(0, 1), real=False)
                    remain -= b

            fid_value = fid.compute().item()
            wandb.log({"eval/fid": fid_value, "epoch": epoch})
            print(f"Epoch {epoch}: FID={fid_value:.3f}")
            model.train()
            if epoch % args.log_images_every == 0 and sample_images:
                grid = make_grid(
                    torch.cat(sample_images, dim=0)[: args.grid_n],
                    nrow=int(math.sqrt(args.grid_n)),
                    padding=2,
                    normalize=True,
                    value_range=(0, 1),
                )
                wandb.log({"viz/samples_grid": wandb.Image(grid), "epoch": epoch})
        ckpt_path = Path(args.save_dir) / f"dit_mnist_epoch{epoch}.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
