import math

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import Tensor

from tread_diffusion.attention import FlexAttentionWithRoPE, RopeKind
from tread_diffusion.typing import typed


@typed
def modulate(
    x: Float[Tensor, "batch seq_len dim"],
    shift: Float[Tensor, "batch dim"],
    scale: Float[Tensor, "batch dim"],
) -> Float[Tensor, "batch seq_len dim"]:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    @typed
    def timestep_embedding(
        t: Float[Tensor, "batch"], dim: int, max_period: int = 10000
    ) -> Float[Tensor, "batch dim"]:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    @typed
    def forward(self, t: Float[Tensor, "batch"]) -> Float[Tensor, "batch hidden_size"]:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    @typed
    def token_drop(
        self,
        labels: Int[Tensor, "batch"],
        force_drop_ids: Int[Tensor, "batch"] | None = None,
    ) -> Int[Tensor, "batch"]:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    @typed
    def forward(
        self,
        labels: Int[Tensor, "batch"],
        train: bool,
        force_drop_ids: Int[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch hidden_size"]:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rope: RopeKind | None = None,
        **block_kwargs,
    ):
        super().__init__()
        self.using_rope = rope is not None
        if rope is not None and rope not in FlexAttentionWithRoPE.ROPE_KINDS:
            raise ValueError("rope must be one of 'axial', 'golden_gate', or None")
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if rope is None:
            self.attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)
        else:
            self.attn = FlexRoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rope_pos_dim=1,
                rope_kind=rope,
                min_freq=0.5,
                max_freq=40.0,
            )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @typed
    def forward(
        self,
        x: Float[Tensor, "batch seq_len hidden_size"],
        c: Float[Tensor, "batch hidden_size"],
        **rope_kwargs,
    ) -> Float[Tensor, "batch seq_len hidden_size"]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), **(rope_kwargs if self.using_rope else {})
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FlexRoPEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rope_pos_dim: int = 1,
        rope_kind: str = "golden_gate",
        min_freq: float = 0.5,
        max_freq: float = 40.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn = FlexAttentionWithRoPE(
            num_heads=num_heads,
            head_dim=head_dim,
            pos_dim=rope_pos_dim,
            min_freq=min_freq,
            max_freq=max_freq,
            rope_kind=rope_kind,
        )

    @typed
    def forward(
        self,
        x: Float[Tensor, "batch seq_len hidden_size"],
        keep_idx: Int[Tensor, "batch seq_len 1"] | None = None,
        original_seq_len: int | None = None,
    ) -> Float[Tensor, "batch seq_len hidden_size"]:
        b, t, c = x.shape
        h = self.num_heads
        qkv = (
            self.qkv(x).view(b, t, 3, h, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        )  # [B,H,T,3,D]
        q, k, v = qkv.unbind(dim=3)  # each [B,H,T,D]
        out = self.attn(q, k, v, keep_idx=keep_idx, original_seq_len=original_seq_len)  # [B,H,T,D]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, h * self.head_dim)
        return self.proj(out)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @typed
    def forward(
        self,
        x: Float[Tensor, "batch num_patches hidden_size"],
        c: Float[Tensor, "batch hidden_size"],
    ) -> Float[Tensor, "batch num_patches unpatched_dim"]:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        rope: RopeKind | None = None,
        route_config: dict | None = None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.rope = rope
        self.use_pos_embed = rope is None
        if self.rope is not None and self.rope not in FlexAttentionWithRoPE.ROPE_KINDS:
            raise ValueError("rope must be one of 'axial', 'golden_gate', or None")

        # Routing config with sane defaults
        default_route = {"start_block": -1, "end_block": -1, "rate": 0.5}
        self.route_config = (
            {**default_route, **{k: v for k, v in (route_config or {}).items() if v is not None}}
            if route_config is not None
            else {}
        )

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size, class_dropout_prob)
        self.null_class = num_classes
        if rope is None:
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, rope=rope) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if hasattr(self, "pos_embed"):
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def token_drop(self, labels, cond_drop_prob, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < cond_drop_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.null_class, labels)
        return labels

    @typed
    def unpatchify(
        self, x: Float[Tensor, "batch num_patches unpatched_dim"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    @typed
    def forward(
        self,
        x: Float[Tensor, "batch in_channels height width"],
        t: Float[Tensor, "batch"],
        y: Int[Tensor, "batch"],
    ) -> Float[Tensor, "batch out_channels height width"]:
        x = self.x_embedder(x)
        if self.use_pos_embed:
            x = x + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        x = self.route_through_blocks(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def route_through_blocks(self, x, c):
        route_indices = None
        if self.route_config:
            start_block = int(self.route_config["start_block"])  # type: ignore[index]
            end_block = int(self.route_config["end_block"])  # type: ignore[index]
            route_rate = float(self.route_config["rate"])  # type: ignore[index]
        else:
            start_block = -1
            end_block = -1
            route_rate = 0.0

        for i, block in enumerate(self.blocks):
            if not self.route_config or i < start_block or i > end_block:
                x = block(x, c)
                continue

            if i == start_block and route_indices is None:
                B, L, C = x.shape
                x_before_routing = x.clone()
                num_to_route = max(1, int(L * route_rate))
                # shuffle L indices between batch
                perms = torch.rand(B, L, device=x.device).argsort(dim=1)
                keep_idx_BL1 = perms[:, :num_to_route].unsqueeze(-1)
                x = torch.take_along_dim(x, keep_idx_BL1, dim=1)

            x = block(x, c, keep_idx=keep_idx_BL1, original_seq_len=L)

            if i == end_block:
                x = x_before_routing.scatter(1, keep_idx_BL1.expand(-1, -1, C), x)
        return x


@typed
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> Float[np.ndarray, "num_tokens embed_dim"]:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


@typed
def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: Float[np.ndarray, "2 1 height width"]
) -> Float[np.ndarray, "num_patches embed_dim"]:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


@typed
def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: Float[np.ndarray, "1 height width"]
) -> Float[np.ndarray, "num_positions embed_dim"]:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


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
