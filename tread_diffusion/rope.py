import os
from abc import ABC

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from tread_diffusion.typing import typed


class RopeImpl(ABC, nn.Module):
    pass


class GoldenGateRoPENd(RopeImpl):
    "https://jerryxio.ng/posts/nd-rope/"

    def __init__(
        self,
        pos_dim: int,
        n_heads: int,
        head_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "freqs_hFP",
            self.make_freqs(head_dim, pos_dim, n_heads, min_freq, max_freq, p_zero_freqs),
        )

    @typed
    def make_freqs(
        self,
        head_dim: int,
        pos_dim: int,
        n_heads: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float,
    ) -> Float[Tensor, "h f p"]:
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)
        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )

        directions_hFP = self.make_directions(n_heads * n_freqs, pos_dim).reshape(
            n_heads, n_freqs, pos_dim
        )
        return directions_hFP * omega_F.reshape(n_freqs, 1)

    @typed
    def _phi(self, m: int) -> float:
        x = 2.0
        for _ in range(10):
            x = (1 + x) ** (1.0 / (m + 1.0))
        return x

    @typed
    def make_directions(self, n: int, d: int) -> Float[Tensor, "n d"]:
        g = self._phi(d)
        alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
        i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
        z = torch.fmod(i * alpha, 1.0)
        directions = torch.erfinv(2.0 * z - 1.0)
        directions = directions / directions.norm(dim=1, keepdim=True)
        return directions.float()

    @typed
    @torch.compile
    def forward(
        self,
        input_NLhd: Float[Tensor, "batch length heads dim"],
        pos_NLP: Float[Tensor, "batch length pos_dim"],
        keep_idx: Int[Tensor, "batch length 1"] | None = None,
    ) -> Float[Tensor, "batch length heads dim"]:
        x_NLhF, y_NLhF = input_NLhd.float().chunk(2, dim=-1)
        theta_NLhF = (self.freqs_hFP * pos_NLP[..., None, None, :].float()).sum(dim=-1)
        cos_NLhF = torch.cos(theta_NLhF)
        sin_NLhF = torch.sin(theta_NLhF)
        if keep_idx is not None:
            keep_idx_NL11 = keep_idx.unsqueeze(-1)
            cos_NLhF = cos_NLhF.take_along_dim(keep_idx_NL11, dim=1)
            sin_NLhF = sin_NLhF.take_along_dim(keep_idx_NL11, dim=1)
        x_out_NLhF = x_NLhF * cos_NLhF - y_NLhF * sin_NLhF
        y_out_NLhF = x_NLhF * sin_NLhF + y_NLhF * cos_NLhF
        output_NLhd = torch.cat((x_out_NLhF, y_out_NLhF), dim=-1)
        return output_NLhd.type_as(input_NLhd)


class AxialRoPE(GoldenGateRoPENd):
    @typed
    def make_directions(self, n: int, d: int) -> Float[Tensor, "n d"]:
        indices = torch.arange(n, dtype=torch.long) % d
        eye = torch.eye(d, dtype=torch.float32)
        return eye.index_select(0, indices)
