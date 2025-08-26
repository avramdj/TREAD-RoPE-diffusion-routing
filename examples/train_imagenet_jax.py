from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils, struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

import wandb
from tread_diffusion_jax import DiT, DiT_models, RectifiedFlow


class ImageNetInt8LatentDataset:
    def __init__(self, data_path: str, labels_path: str):
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        num_samples = len(self.labels)
        self.data = np.memmap(data_path, dtype="uint8", mode="r", shape=(num_samples, 4096))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        vec = self.data[idx]
        x = vec.astype(np.float32).reshape(4, 32, 32)
        x = (x / 255.0 - 0.5) * 24.0
        label_idx = int(self.labels[idx][0])
        return x, label_idx


def shard_batch(batch: np.ndarray) -> np.ndarray:
    devices = jax.local_device_count()
    b = batch.shape[0]
    assert b % devices == 0, f"Batch {b} not divisible by devices {devices}"
    return batch.reshape(devices, b // devices, *batch.shape[1:])


def data_iterator(
    ds: ImageNetInt8LatentDataset, *, batch_size: int, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(ds))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(ds) - batch_size + 1, batch_size):
            idx = indices[start : start + batch_size]
            xs = []
            ys = []
            for i in idx:
                x, y = ds[i]
                xs.append(x)
                ys.append(y)
            batch_x = np.stack(xs, axis=0)
            batch_y = np.array(ys, dtype=np.int32)
            yield batch_x, batch_y


@struct.dataclass
class TrainStateEMA(TrainState):
    ema_params: FrozenDict


def create_train_state(rng, model: DiT, learning_rate: float, ema_decay: float):
    variables = model.init(
        rng,
        jnp.zeros((1, 4, 32, 32), jnp.float32),
        jnp.zeros((1,), jnp.float32),
        jnp.zeros((1,), jnp.int32),
        train=True,
    )
    params = variables["params"]
    tx = optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.999, weight_decay=1e-3)
    state = TrainStateEMA.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    return state, variables


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiT (JAX) on ImageNet.int8 latents")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="examples/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-n", type=int, default=16)
    parser.add_argument("--fid-samples", type=int, default=256)
    parser.add_argument("--dit-size", type=str, default="DiT-S/2")
    parser.add_argument("--rope", type=str, default="none", choices=["none", "axial", "golden_gate"])
    parser.add_argument("--inet-data", type=str, default="examples/data/imagenet_int8/inet.npy")
    parser.add_argument("--inet-labels", type=str, default="examples/data/imagenet_int8/inet.json")
    parser.add_argument("--wandb-name", type=str, default="tread-diffusion-imagenet-int8-jax")
    parser.add_argument("--save-name", type=str, default="dit-jax-latest")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    args = parser.parse_args()

    # Setup
    rng = jax.random.PRNGKey(args.seed)
    wandb_project = "tread-diffusion-imagenet-int8-jax"
    wandb.init(
        project=wandb_project,
        name=args.wandb_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timesteps": args.timesteps,
        },
    )

    # Data
    ds = ImageNetInt8LatentDataset(args.inet_data, args.inet_labels)
    it = data_iterator(ds, batch_size=args.batch_size, seed=args.seed)

    # Model & RF
    rope_arg = None if args.rope == "none" else args.rope
    dit_args = {
        "input_size": 32,
        "in_channels": 4,
        "num_classes": 1000,
        "learn_sigma": False,
        "rope": rope_arg,
    }
    model = DiT_models[args.dit_size](**dit_args)
    state, variables = create_train_state(rng, model, args.lr, args.ema_decay)
    rf = RectifiedFlow(height=32, width=32, num_steps=args.timesteps, vae_scaling_factor=1.0)
    num_devices = jax.local_device_count()
    assert args.batch_size % num_devices == 0, "Batch size must be divisible by local device count"

    def update_ema(ema, params):
        return jax.tree_util.tree_map(
            lambda e_, p_: e_ * args.ema_decay + (1.0 - args.ema_decay) * p_, ema, params
        )

    @jax.pmap(axis_name="devices")
    def train_step(
        state: TrainStateEMA,
        variables: dict,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        step_rng: jax.Array,
    ):
        def loss_fn(params):
            vars_all = {**variables, "params": params}
            loss = rf.loss(vars_all, model.apply, batch_x, batch_y, step_rng, train=True)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        new_state = state.apply_gradients(grads=grads)
        new_ema = update_ema(new_state.ema_params, new_state.params)
        new_state = new_state.replace(ema_params=new_ema)
        return new_state, loss

    # Replicate state/variables across devices
    state = jax_utils.replicate(state)
    variables_repl = {k: jax_utils.replicate(v) for k, v in variables.items()}

    # Training
    total_steps = args.epochs * (len(ds) // args.batch_size)
    for step in range(1, total_steps + 1):
        batch_x_np, batch_y_np = next(it)
        batch_x = shard_batch(batch_x_np)
        batch_y = shard_batch(batch_y_np)
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_devices)
        state, loss = train_step(
            state, variables_repl, jnp.asarray(batch_x), jnp.asarray(batch_y), step_rngs
        )
        if step % 10 == 0:
            loss_scalar = float(jax.device_get(loss)[0])
            wandb.log({"train/loss": loss_scalar, "step": step})

    # Save params
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.save_dir) / f"{args.save_name}.npz"
    from flax.serialization import to_state_dict

    state_host = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    state_dict = to_state_dict({"params": state_host.params, "ema_params": state_host.ema_params})
    np.savez(ckpt_path, **state_dict)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
