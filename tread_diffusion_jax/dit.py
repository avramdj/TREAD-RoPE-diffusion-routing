from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


def _silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256

    @staticmethod
    def timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10000) -> jnp.ndarray:
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size)(t_freq)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size)(x)
        return x


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float

    @nn.compact
    def __call__(
        self, labels: jnp.ndarray, train: bool, force_drop_ids: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        use_cfg_embedding = self.dropout_prob > 0.0
        emb = nn.Embed(
            num_embeddings=self.num_classes + int(use_cfg_embedding),
            features=self.hidden_size,
            name="embedding_table",
        )

        def token_drop(labels_in: jnp.ndarray) -> jnp.ndarray:
            if force_drop_ids is None:
                drop = jax.random.bernoulli(
                    nn.make_rng("dropout"), p=self.dropout_prob, shape=labels_in.shape
                )
            else:
                drop = force_drop_ids == 1
            return jnp.where(drop, self.num_classes, labels_in)

        if (train and self.dropout_prob > 0.0) or (force_drop_ids is not None):
            labels = token_drop(labels)
        return emb(labels)


class DiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    rope: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        c: jnp.ndarray,
        *,
        keep_idx: Optional[jnp.ndarray] = None,
        original_seq_len: Optional[int] = None,
    ) -> jnp.ndarray:
        from .attention import FlexAttentionWithRoPE

        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        ada = nn.Sequential([nn.silu, nn.Dense(6 * self.hidden_size)])

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(ada(c), 6, axis=1)

        x_attn = norm1(x)
        x_attn = x_attn * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        if self.rope is None:
            attn = nn.SelfAttention(
                num_heads=self.num_heads,
                dtype=x.dtype,
                qkv_features=self.hidden_size,
                out_features=self.hidden_size,
                use_bias=True,
                broadcast_dropout=False,
                dropout_rate=0.0,
                deterministic=True,
            )
            x = x + gate_msa[:, None, :] * attn(x_attn)
        else:
            attn = FlexAttentionWithRoPE(
                num_heads=self.num_heads,
                head_dim=self.hidden_size // self.num_heads,
                rope_kind=self.rope,
            )
            x = x + gate_msa[:, None, :] * attn(
                x_attn, keep_idx=keep_idx, original_seq_len=original_seq_len
            )

        x_mlp = norm2(x)
        x_mlp = x_mlp * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        y = nn.Dense(mlp_hidden)(x_mlp)
        y = nn.gelu(y, approximate=True)
        y = nn.Dense(self.hidden_size)(y)
        x = x + gate_mlp[:, None, :] * y
        return x


class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        norm = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        ada = nn.Sequential([nn.silu, nn.Dense(2 * self.hidden_size)])
        shift, scale = jnp.split(ada(c), 2, axis=1)
        x = norm(x)
        x = x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels)(x)
        return x


class PatchEmbed(nn.Module):
    input_size: int
    patch_size: int
    in_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=True,
        )
        x = conv(x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        return x


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> jnp.ndarray:
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.stack(jnp.meshgrid(grid_w, grid_h, indexing="xy"), axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: jnp.ndarray) -> jnp.ndarray:
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


@dataclass
class RouteConfig:
    start_block: int = -1
    end_block: int = -1
    rate: float = 0.0
    mix_factor: float = 0.0


class DiT(nn.Module):
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = False
    route_config: Optional[Dict[str, Any]] = None
    rope: Optional[str] = None

    def setup(self) -> None:
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.x_embedder = PatchEmbed(
            self.input_size, self.patch_size, self.in_channels, self.hidden_size
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes + 1, self.hidden_size, self.class_dropout_prob)
        self.null_class = self.num_classes

        num_patches_side = self.input_size // self.patch_size
        num_patches = num_patches_side * num_patches_side
        pos = _get_2d_sincos_pos_embed(self.hidden_size, num_patches_side)
        self.pos_embed = self.param(
            "pos_embed",
            lambda *_: pos.astype(jnp.float32)[None, ...],
            (1, num_patches, self.hidden_size),
        )

        self.blocks = [
            DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, rope=self.rope)
            for _ in range(self.depth)
        ]
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

    def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
        c = self.out_channels
        p = self.patch_size
        h = w = int(jnp.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = jnp.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        imgs = jnp.transpose(imgs, (0, 2, 3, 1))
        return imgs

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.x_embedder(x)
        if self.rope is None:
            x = x + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, train)
        c = t_emb + y_emb
        # Routing like PyTorch impl
        start_block = int(self.route_config.get("start_block", -1)) if self.route_config else -1
        end_block = int(self.route_config.get("end_block", -1)) if self.route_config else -1
        route_rate = float(self.route_config.get("rate", 0.0)) if self.route_config else 0.0
        keep_idx = None
        original_seq_len = None
        x_before = None
        for i, block in enumerate(self.blocks):
            if not self.route_config or i < start_block or i > end_block:
                x = block(x, c)
                continue
            if i == start_block and keep_idx is None:
                B, L, Cblk = x.shape
                original_seq_len = L
                x_before = x
                num_to_route = max(1, int(L * route_rate))
                perms = jax.random.permutation(nn.make_rng("dropout"), L, independent=True)
                perms = jnp.tile(perms[None, :], (B, 1))
                keep_idx = perms[:, :num_to_route][:, :, None]
                x = jnp.take_along_axis(x, keep_idx.repeat(Cblk, axis=2), axis=1)
            x = block(x, c, keep_idx=keep_idx, original_seq_len=original_seq_len)
            if i == end_block and x_before is not None and keep_idx is not None:
                x = x_before.at[jnp.arange(x.shape[0])[:, None], keep_idx.squeeze(-1)].set(x)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


def _dit(depth: int, hidden_size: int, patch_size: int, num_heads: int, **kwargs: Any) -> DiT:
    return DiT(
        depth=depth, hidden_size=hidden_size, patch_size=patch_size, num_heads=num_heads, **kwargs
    )


def DiT_XL_2(**kwargs: Any) -> DiT:
    return _dit(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs: Any) -> DiT:
    return _dit(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs: Any) -> DiT:
    return _dit(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs: Any) -> DiT:
    return _dit(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs: Any) -> DiT:
    return _dit(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs: Any) -> DiT:
    return _dit(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs: Any) -> DiT:
    return _dit(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs: Any) -> DiT:
    return _dit(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs: Any) -> DiT:
    return _dit(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs: Any) -> DiT:
    return _dit(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs: Any) -> DiT:
    return _dit(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs: Any) -> DiT:
    return _dit(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
