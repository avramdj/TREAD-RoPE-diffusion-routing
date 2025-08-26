from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class GoldenGateRoPENd(nn.Module):
    pos_dim: int
    n_heads: int
    head_dim: int
    min_freq: float
    max_freq: float
    p_zero_freqs: float = 0.0

    def setup(self) -> None:
        def make_freqs() -> jnp.ndarray:
            n_freqs = self.head_dim // 2
            n_zero_freqs = int(round(self.p_zero_freqs * n_freqs))
            zeros = jnp.zeros((n_zero_freqs,), dtype=jnp.float32)
            rest = self.min_freq * (
                (self.max_freq / self.min_freq) ** jnp.linspace(0.0, 1.0, n_freqs - n_zero_freqs)
            )
            omega_F = jnp.concatenate([zeros, rest], axis=0)  # [F]

            directions = self.make_directions(self.n_heads * n_freqs, self.pos_dim).reshape(
                self.n_heads, n_freqs, self.pos_dim
            )
            return directions * omega_F[:, None]

        self.freqs_hFP = self.variable("constants", "freqs_hFP", make_freqs)

    @staticmethod
    def _phi(m: int) -> float:
        x = 2.0
        for _ in range(10):
            x = (1.0 + x) ** (1.0 / (m + 1.0))
        return float(x)

    @classmethod
    def make_directions(cls, n: int, d: int) -> jnp.ndarray:
        g = cls._phi(d)
        alpha = (1.0 / g) ** jnp.arange(1, d + 1, dtype=jnp.float32)  # [d]
        i = jnp.arange(1, n + 1, dtype=jnp.float32)[:, None]  # [n,1]
        z = jnp.mod(i * alpha[None, :], 1.0)
        directions = jax.scipy.special.erfinv(2.0 * z - 1.0)
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        return directions.astype(jnp.float32)

    def __call__(
        self,
        input_BLhd: jnp.ndarray,
        pos_BLP: jnp.ndarray,
        *,
        keep_idx: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # input: [B,L,H,D], pos: [B,L,P]
        x, y = jnp.split(input_BLhd.astype(jnp.float32), 2, axis=-1)  # [B,L,H,F]
        freqs_hFP = self.freqs_hFP.value  # [H,F,P]
        # broadcast to [B,L,H,F]
        theta = jnp.einsum("hfp,blp->blhf", freqs_hFP, pos_BLP.astype(jnp.float32))
        if keep_idx is not None:
            # keep_idx: [B,Lk,1] -> index along L
            theta = jnp.take_along_axis(theta, keep_idx, axis=1)
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        if keep_idx is not None:
            x = jnp.take_along_axis(x, keep_idx, axis=1)
            y = jnp.take_along_axis(y, keep_idx, axis=1)
        x_out = x * cos - y * sin
        y_out = x * sin + y * cos
        return jnp.concatenate([x_out, y_out], axis=-1).astype(input_BLhd.dtype)


class AxialRoPE(GoldenGateRoPENd):
    @classmethod
    def make_directions(cls, n: int, d: int) -> jnp.ndarray:
        indices = jnp.arange(n, dtype=jnp.int32) % d
        eye = jnp.eye(d, dtype=jnp.float32)
        return eye[indices]
