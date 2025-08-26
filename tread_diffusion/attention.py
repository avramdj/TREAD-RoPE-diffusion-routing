from __future__ import annotations

import os
from typing import Callable, Final, Literal, Optional, get_args

import torch
import torch.nn as nn
from jaxtyping import Int
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from tread_diffusion.rope import AxialRoPE, GoldenGateRoPENd

RopeKind = Literal["axial", "golden_gate"]

FLEX_ATTNS = {
    "cpu": _flex_attention,
}

if torch.cuda.is_available():
    if os.getenv("FLEX_ATTN_NO_COMPILE", "").lower() in {"1", "true", "yes"}:
        FLEX_ATTNS["cuda"] = _flex_attention
    else:
        FLEX_ATTNS["cuda"] = torch.compile(_flex_attention)


class FlexAttentionWithRoPE(nn.Module):
    """FlexAttention + RoPE. Inputs: [B,H,L,D]."""

    ROPE_KINDS: Final[tuple[RopeKind, ...]] = get_args(RopeKind)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        pos_dim: int = 2,
        min_freq: float = 0.5,
        max_freq: float = 40.0,
        p_zero_freqs: float = 0.0,
        rope_kind: RopeKind = "golden_gate",
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        if rope_kind not in self.ROPE_KINDS:
            raise ValueError(f"rope_kind must be one of {self.ROPE_KINDS}")

        rope_cls: type[GoldenGateRoPENd]
        if rope_kind == "axial":
            rope_cls = AxialRoPE
        elif rope_kind == "golden_gate":
            rope_cls = GoldenGateRoPENd
        else:
            raise ValueError(f"Unknown rope_kind: {rope_kind}")

        self.rope = rope_cls(
            pos_dim=pos_dim,
            n_heads=num_heads,
            head_dim=head_dim,
            min_freq=min_freq,
            max_freq=max_freq,
            p_zero_freqs=p_zero_freqs,
        )

        def _noop(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            return score

        self._default_score_mod: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] = _noop

    @staticmethod
    def _ensure_pos(
        q: Tensor,
        pos: Optional[Tensor],
        *,
        pos_dim: int,
        original_seq_len: Optional[int] = None,
    ) -> Tensor:
        """Default 1D positions if none provided. q=[B,H,L,D] -> pos=[B,L,P]."""
        if pos is not None:
            return pos
        batch_size, _, seq_len, _ = q.shape
        seq_len = original_seq_len if original_seq_len is not None else seq_len
        device = q.device
        positions_1d = torch.arange(seq_len, device=device, dtype=torch.float32)
        if pos_dim == 1:
            pos_blp = positions_1d.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        else:
            pos_blp = (
                positions_1d.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1).repeat(1, 1, pos_dim)
            )
        return pos_blp

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        pos: Optional[Tensor] = None,
        score_mod: Optional[Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]] = None,
        keep_idx: Optional[Int[Tensor, "batch seq_len 1"]] = None,
        original_seq_len: Optional[int] = None,
    ) -> Tensor:
        """Returns [B,H,L,D]. Applies RoPE to Q,K then FlexAttention."""
        device = query.device
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query/key/value must be rank-4: [batch, heads, seq_len, head_dim]")

        if not (query.shape == key.shape == value.shape):
            raise ValueError("query, key, value must have identical shapes")

        batch_size, num_heads, seq_len, head_dim = query.shape

        q_blh_d = query.permute(0, 2, 1, 3).contiguous()
        k_blh_d = key.permute(0, 2, 1, 3).contiguous()

        rope_pos_dim: int = self.rope.freqs_hFP.shape[-1]
        pos_blp = self._ensure_pos(query, pos, pos_dim=rope_pos_dim, original_seq_len=original_seq_len)

        q_rot = self.rope(q_blh_d, pos_blp, keep_idx=keep_idx)
        k_rot = self.rope(k_blh_d, pos_blp, keep_idx=keep_idx)

        q_bhld = q_rot.permute(0, 2, 1, 3).contiguous()
        k_bhld = k_rot.permute(0, 2, 1, 3).contiguous()

        mod = score_mod if score_mod is not None else self._default_score_mod

        out_bhld = FLEX_ATTNS[device.type](q_bhld, k_bhld, value, score_mod=mod)
        return out_bhld


__all__ = [
    "RopeKind",
    "FlexAttentionWithRoPE",
]
