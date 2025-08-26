from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .rope import AxialRoPE, GoldenGateRoPENd


class FlexAttentionWithRoPE(nn.Module):
    num_heads: int
    head_dim: int
    pos_dim: int = 1
    min_freq: float = 0.5
    max_freq: float = 40.0
    p_zero_freqs: float = 0.0
    rope_kind: str = "golden_gate"

    def setup(self) -> None:
        rope_cls = GoldenGateRoPENd if self.rope_kind == "golden_gate" else AxialRoPE
        self.rope = rope_cls(
            pos_dim=self.pos_dim,
            n_heads=self.num_heads,
            head_dim=self.head_dim,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            p_zero_freqs=self.p_zero_freqs,
        )

    @staticmethod
    def _ensure_pos(
        q_bhld: jnp.ndarray,
        pos_blp: Optional[jnp.ndarray],
        *,
        pos_dim: int,
        original_seq_len: Optional[int] = None,
    ) -> jnp.ndarray:
        if pos_blp is not None:
            return pos_blp
        b, h, l, d = q_bhld.shape
        seq_len = original_seq_len if original_seq_len is not None else l
        positions_1d = jnp.arange(seq_len, dtype=jnp.float32)
        if pos_dim == 1:
            return positions_1d[None, :].repeat(b, axis=0)[:, :, None]
        return jnp.tile(positions_1d[None, :, None], (b, 1, pos_dim))

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        keep_idx: Optional[jnp.ndarray] = None,
        original_seq_len: Optional[int] = None,
    ) -> jnp.ndarray:
        # x: [B,L,C] -> qkv, apply rope on q/k, flex attention, return [B,L,C]
        c = x.shape[-1]
        h = self.num_heads
        qkv = nn.Dense(3 * c)(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, h, self.head_dim).transpose(0, 3, 1, 2, 4)
        q, k, v = jnp.split(qkv, 3, axis=3)
        q = q.squeeze(3)
        k = k.squeeze(3)
        v = v.squeeze(3)  # [B,H,L,D]
        pos_blp = self._ensure_pos(
            q, None, pos_dim=self.rope.pos_dim, original_seq_len=original_seq_len
        )
        q_rot = self.rope(jnp.transpose(q, (0, 2, 1, 3)), pos_blp, keep_idx=keep_idx)  # [B,L,H,D]
        k_rot = self.rope(jnp.transpose(k, (0, 2, 1, 3)), pos_blp, keep_idx=keep_idx)
        q_bhld = jnp.transpose(q_rot, (0, 2, 1, 3))
        k_bhld = jnp.transpose(k_rot, (0, 2, 1, 3))

        attn_logits = jnp.einsum("bhld,bhmd->bhlm", q_bhld, k_bhld) / jnp.sqrt(self.head_dim)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("bhlm,bhmd->bhld", attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], h * self.head_dim)
        out = nn.Dense(c)(out)
        return out
