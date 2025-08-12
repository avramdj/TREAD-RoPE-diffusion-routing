from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb
from tread_diffusion import DiT, DiT_models


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
    s1 = sqrt_ac.gather(0, t).view(-1, 1, 1, 1)
    s2 = sqrt_1m_ac.gather(0, t).view(-1, 1, 1, 1)
    return s1 * x0 + s2 * noise


@torch.no_grad()
def sample_model(
    model: DiT,
    *,
    num_images: int,
    device: torch.device,
    num_steps: int,
    classes: Tensor | None = None,
    height: int = 28,
    width: int = 28,
) -> Tensor:
    return model.sample_ode_rectified(
        num_images,
        height=height,
        width=width,
        device=device,
        num_steps=num_steps,
        class_labels=classes.long() if classes is not None else None,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train DiT on MNIST (pixel space)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--rope", type=str, default="none", choices=["none", "axial", "golden_gate"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="examples/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-samples", type=int, default=256)
    parser.add_argument("--log-fid-every", type=int, default=100)
    parser.add_argument("--log-images-every", type=int, default=10)
    parser.add_argument("--grid-n", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

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

    # Small dit (hopeflly sane defaults for this depth)
    rope_arg = None if args.rope == "none" else args.rope
    route_config = {"start_block": 2, "end_block": 10, "rate": 0.5, "mix_factor": 0.5}
    model = DiT_models["DiT-S/2"](
        input_size=28,
        in_channels=1,
        num_classes=10,
        learn_sigma=False,
        rope=rope_arg,
        route_config=route_config,
    ).to(device)

    from torch._inductor import config as inductor_config

    torch.set_float32_matmul_precision("high")
    inductor_config.triton.cudagraphs = False
    model = torch.compile(model, dynamic=True, fullgraph=False, mode="max-autotune-no-cudagraphs")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # RECTIFIED FLOW

    wandb_project = os.getenv("WANDB_PROJECT", "tread-diffusion-mnist")
    wandb.init(
        project=wandb_project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timesteps": args.timesteps,
            "rope": args.rope,
            "route_config": route_config,
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

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        num_batches = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            x1 = torch.randn_like(images)
            t = torch.rand(images.shape[0], device=device)
            x_t = (1.0 - t).view(-1, 1, 1, 1) * x1 + t.view(-1, 1, 1, 1) * images
            v_target = images - x1

            with amp_ctx:
                pred_v = model(x_t, t, labels)
                loss = F.mse_loss(pred_v, v_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            tqdm.write("") if False else None

        avg = running_loss / max(1, num_batches)
        wandb.log({"train/loss": avg, "epoch": epoch})
        print(f"Epoch {epoch}: loss={avg:.4f}")

        # FID evaluation
        if epoch % args.log_fid_every == 0:
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            model.eval()
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

                n = min(args.fid_samples, args.batch_size)
                remain = args.fid_samples
                sample_images = []
                while remain > 0:
                    b = min(remain, n)
                    y = torch.randint(0, 10, (b,), device=device)
                    x_t = sample_model(
                        model,
                        num_images=b,
                        device=device,
                        num_steps=args.timesteps,
                        classes=y,
                    )
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

        # Log images
        if epoch % args.log_images_every == 0:
            model.eval()
            with torch.no_grad():
                b = args.grid_n
                y = torch.randint(0, 10, (b,), device=device)
                x_t = sample_model(
                    model,
                    num_images=b,
                    device=device,
                    num_steps=args.timesteps,
                    classes=y,
                )
                imgs = (x_t + 1.0) * 0.5
                grid = make_grid(
                    imgs.detach().cpu()[: args.grid_n],
                    nrow=int(math.sqrt(args.grid_n)),
                    padding=2,
                    normalize=True,
                    value_range=(0, 1),
                )
                wandb.log({"viz/samples_grid": wandb.Image(grid), "epoch": epoch})
            model.train()
        ckpt_path = Path(args.save_dir) / f"dit_mnist_epoch{epoch}.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
