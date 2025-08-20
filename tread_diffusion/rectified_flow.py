from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torchdiffeq import odeint

from tread_diffusion.apg import MomentumBuffer, adaptive_projected_guidance
from tread_diffusion.typing import typed


class RectifiedFlow:
    def __init__(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        num_steps: int = 50,
        ode_method: str = "rk4",
        vae_scaling_factor: float = 1.0,
    ) -> None:
        self.height = height
        self.width = width
        self.num_steps = num_steps
        self.ode_method = ode_method
        self.vae_scaling_factor = vae_scaling_factor

    @typed
    def loss(
        self,
        model: nn.Module,
        x0: Float[Tensor, "batch in_channels height width"],
        y: Int[Tensor, "batch"],
        amp_ctx: torch.amp.autocast_mode.autocast | nullcontext = nullcontext,
    ) -> Float[Tensor, ""]:
        x0 = x0 * self.vae_scaling_factor
        x1 = torch.randn_like(x0)
        t = torch.rand(x0.shape[0], device=x0.device)
        x_t = (1.0 - t).view(-1, 1, 1, 1) * x1 + t.view(-1, 1, 1, 1) * x0
        v_target = x0 - x1
        with amp_ctx:
            pred_v = model(x_t, t, y)
        return F.mse_loss(pred_v.float(), v_target.float())

    @typed
    @torch.no_grad()
    def sample_diffeq(
        self,
        num_images: int,
        *,
        model: nn.Module,
        height: int | None = None,
        width: int | None = None,
        num_steps: int | None = None,
        class_labels: Int[Tensor, "batch"] | None = None,
        method: str | None = None,
        cfg_scale: float | None = None,
        apg: bool = False,
        full_tokens: bool = True,
    ) -> Float[Tensor, "batch in_channels height width"]:
        """Rectified Flow ODE sampling using torchdiffeq."""
        orig_route_config = model.route_config
        if full_tokens:
            model.route_config = None

        null_class = model.null_class
        height = height or self.height
        width = width or self.width
        if height is None or width is None:
            raise ValueError("height and width must be provided either at init or call time")
        num_steps = num_steps or self.num_steps
        method = method or self.ode_method

        device = next(model.parameters()).device
        batch_size = num_images
        channels = model.in_channels
        x1 = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
        if class_labels is None:
            num_classes = getattr(model.y_embedder, "num_classes", None)
            if num_classes is None:
                raise ValueError("num_classes is not set")
            class_labels = torch.randint(0, int(num_classes), (batch_size,), device=device)

        class ODEFunc(nn.Module):
            def __init__(
                self,
                outer: nn.Module,
                labels: Tensor,
                null_labels: Tensor | None,
                cfg_scale: float | None,
                apg: bool,
            ):
                super().__init__()
                self.outer = outer
                self.labels = labels
                self.null_labels = null_labels
                self.cfg_scale = cfg_scale
                self.apg = apg
                self.momentum_buffer = MomentumBuffer()

            def forward(self, t: Tensor, x: Tensor) -> Tensor:
                t_vec = t.expand(x.shape[0]).to(x)
                with torch.no_grad():
                    v = self.outer(x, t_vec, self.labels)
                    if self.null_labels is not None and self.cfg_scale is not None:
                        vnull = self.outer(x, t_vec, self.null_labels)
                        if self.apg:
                            v = adaptive_projected_guidance(
                                v, vnull, self.cfg_scale, self.momentum_buffer
                            )
                        else:
                            dd = v - vnull
                            v = v + self.cfg_scale * dd
                return v

        if null_class is not None and cfg_scale is not None:
            null_labels = torch.full_like(class_labels, null_class)
        else:
            null_labels = None

        func = ODEFunc(model, class_labels, null_labels, cfg_scale, apg)
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
        x_path = odeint(func, x1, ts, method=method)
        x0 = x_path[-1]
        model.route_config = orig_route_config
        return x0 / self.vae_scaling_factor

    @typed
    @torch.no_grad()
    def sample_euler(
        self,
        num_images: int,
        *,
        model: nn.Module,
        height: int | None = None,
        width: int | None = None,
        num_steps: int | None = None,
        class_labels: Int[Tensor, "batch"] | None = None,
        cfg_scale: float | None = None,
        apg: bool = False,
        full_tokens: bool = True,
    ) -> Float[Tensor, "batch in_channels height width"]:
        """Rectified Flow ODE sampling using explicit Euler integration."""
        orig_route_config = model.route_config
        if full_tokens:
            model.route_config = None

        null_class = model.null_class
        height = height or self.height
        width = width or self.width
        if height is None or width is None:
            raise ValueError("height and width must be provided either at init or call time")
        num_steps = num_steps or self.num_steps

        device = next(model.parameters()).device
        batch_size = num_images
        channels = model.in_channels
        x1 = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
        if class_labels is None:
            num_classes = getattr(model.y_embedder, "num_classes", None)
            if num_classes is None:
                raise ValueError("num_classes is not set")
            class_labels = torch.randint(0, int(num_classes), (batch_size,), device=device)

        if null_class is not None and cfg_scale is not None:
            null_labels = torch.full_like(class_labels, null_class)
        else:
            null_labels = None

        ts = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=torch.float32)

        momentum_buffer = MomentumBuffer()

        for i in range(num_steps):
            t_now = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_now
            t_vec = t_now.expand(batch_size).to(torch.float32)
            v_pred = model(x1.to(torch.float32), t_vec, class_labels).float()
            if null_labels is not None and cfg_scale is not None:
                vnull = model(x1.to(torch.float32), t_vec, null_labels).float()
                if apg:
                    v_pred = adaptive_projected_guidance(v_pred, vnull, cfg_scale, momentum_buffer)
                else:
                    dd = v_pred - vnull
                    v_pred = v_pred + cfg_scale * dd
            x1 = x1 + v_pred * dt
        model.route_config = orig_route_config
        return x1 / self.vae_scaling_factor
