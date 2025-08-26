from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp


def _normalize_along_channels_hw(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    # x: [B,C,H,W]
    norm = jnp.linalg.norm(x, axis=(1, 2, 3), keepdims=True)
    norm = jnp.maximum(norm, eps)
    return x / norm


def _project(v0: jnp.ndarray, v1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Both [B,C,H,W]
    v1_unit = _normalize_along_channels_hw(v1)
    v0_parallel = (jnp.sum(v0 * v1_unit, axis=(1, 2, 3), keepdims=True)) * v1_unit
    v0_orth = v0 - v0_parallel
    return v0_parallel, v0_orth


def _adaptive_projected_guidance(
    pred_cond: jnp.ndarray,
    pred_uncond: jnp.ndarray,
    guidance_scale: float,
    running_average: jnp.ndarray,
    *,
    momentum: float = -0.75,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    diff = pred_cond - pred_uncond
    new_avg = momentum * running_average + diff
    if norm_threshold > 0:
        diff_norm = jnp.linalg.norm(new_avg, axis=(1, 2, 3), keepdims=True)
        scale_factor = jnp.minimum(1.0, norm_threshold / (diff_norm + 1e-8))
        new_avg = new_avg * scale_factor
    diff_parallel, diff_orth = _project(new_avg, pred_cond)
    normalized_update = diff_orth + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1.0) * normalized_update
    return pred_guided, new_avg


@dataclass
class RectifiedFlow:
    height: Optional[int] = None
    width: Optional[int] = None
    num_steps: int = 50
    vae_scaling_factor: float = 1.0

    def loss(
        self, variables, model_apply, x0: jnp.ndarray, y: jnp.ndarray, rng: jax.Array, *, train: bool
    ) -> jnp.ndarray:
        x0 = x0 * self.vae_scaling_factor
        n_key, t_key = jax.random.split(rng)
        x1 = jax.random.normal(n_key, x0.shape, dtype=jnp.float32)
        t = jax.random.uniform(t_key, (x0.shape[0],), minval=0.0, maxval=1.0)
        t_b = t[:, None, None, None]
        x_t = (1.0 - t_b) * x1 + t_b * x0
        v_target = x0 - x1
        pred_v = model_apply(variables, x_t, t, y, train=train, rngs={"dropout": rng})
        return jnp.mean((pred_v.astype(jnp.float32) - v_target.astype(jnp.float32)) ** 2)

    def sample_euler(
        self,
        variables,
        model_apply,
        *,
        rng: jax.Array,
        num_images: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_steps: Optional[int] = None,
        class_labels: Optional[jnp.ndarray] = None,
        cfg_scale: Optional[float] = None,
        apg: bool = False,
    ) -> jnp.ndarray:
        h = height or self.height
        w = width or self.width
        if h is None or w is None:
            raise ValueError("height and width must be provided either at init or call time")
        steps = int(num_steps or self.num_steps)

        b_rng, t_rng = jax.random.split(rng)
        kernel = variables["params"]["x_embedder"]["Conv_0"]["kernel"]
        channels = int(kernel.shape[2])
        x = jax.random.normal(b_rng, (num_images, channels, h, w), dtype=jnp.float32)
        emb = variables["params"]["y_embedder"]["embedding_table"]["embedding"]
        num_classes = int(emb.shape[0] - 1)
        if class_labels is None:
            class_labels = jax.random.randint(t_rng, (num_images,), minval=0, maxval=num_classes)
        null_labels = jnp.full_like(class_labels, num_classes) if cfg_scale is not None else None

        ts = jnp.linspace(0.0, 1.0, steps + 1, dtype=jnp.float32)
        m0 = jnp.zeros_like(x)

        def body_fn(carry, i):
            x_curr, m_curr, step_key = carry
            key_i, step_key = jax.random.split(step_key)
            t_now = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_now
            t_vec = jnp.full((num_images,), t_now, dtype=jnp.float32)
            v = model_apply(
                variables, x_curr, t_vec, class_labels, train=False, rngs={"dropout": key_i}
            )
            if cfg_scale is not None and null_labels is not None:
                vnull = model_apply(
                    variables, x_curr, t_vec, null_labels, train=False, rngs={"dropout": key_i}
                )
                if apg:
                    v, m_next = _adaptive_projected_guidance(v, vnull, float(cfg_scale), m_curr)
                else:
                    dd = v - vnull
                    v = v + float(cfg_scale) * dd
                    m_next = m_curr
            else:
                m_next = m_curr
            x_next = x_curr + v.astype(jnp.float32) * dt
            return (x_next, m_next, step_key), None

        (x_out, _, _), _ = jax.lax.scan(body_fn, (x, m0, rng), jnp.arange(steps))
        return x_out / self.vae_scaling_factor

    def sample_diffeq(
        self,
        variables,
        model_apply,
        *,
        rng: jax.Array,
        num_images: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_steps: Optional[int] = None,
        class_labels: Optional[jnp.ndarray] = None,
        cfg_scale: Optional[float] = None,
        apg: bool = False,
    ) -> jnp.ndarray:
        h = height or self.height
        w = width or self.width
        if h is None or w is None:
            raise ValueError("height and width must be provided either at init or call time")
        steps = int(num_steps or self.num_steps)

        b_rng, t_rng = jax.random.split(rng)
        kernel = variables["params"]["x_embedder"]["Conv_0"]["kernel"]
        channels = int(kernel.shape[2])
        x1 = jax.random.normal(b_rng, (num_images, channels, h, w), dtype=jnp.float32)

        emb = variables["params"]["y_embedder"]["embedding_table"]["embedding"]
        num_classes = int(emb.shape[0] - 1)
        if class_labels is None:
            class_labels = jax.random.randint(t_rng, (num_images,), minval=0, maxval=num_classes)
        null_labels = jnp.full_like(class_labels, num_classes) if cfg_scale is not None else None

        def vfield(t, y, args):
            t_vec = jnp.full((num_images,), t, dtype=jnp.float32)
            v = model_apply(variables, y, t_vec, class_labels, train=False, rngs={"dropout": rng})
            if cfg_scale is not None and null_labels is not None:
                vnull = model_apply(
                    variables, y, t_vec, null_labels, train=False, rngs={"dropout": rng}
                )
                if apg:
                    diff = v - vnull
                    diff_par, diff_orth = _project(diff, v)
                    v = v + (float(cfg_scale) - 1.0) * (diff_orth)
                else:
                    dd = v - vnull
                    v = v + float(cfg_scale) * dd
            return v.astype(jnp.float32)

        term = dfx.ODETerm(vfield)
        solver = dfx.Dopri5()
        t0 = 0.0
        t1 = 1.0
        saveat = dfx.SaveAt(t1=True)
        stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-3)
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=(t1 - t0) / steps,
            y0=x1,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        x0 = sol.ys
        return x0 / self.vae_scaling_factor
