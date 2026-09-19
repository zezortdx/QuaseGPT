"""Training loop (plain PyTorch, no HF Trainer). Same code runs locally or in Colab.

Usage:
  python ml/train.py --config configs/training.json
  python ml/train.py --config configs/training.json --resume /path/to/checkpoint.pt

Local CPU training works but is slow; Colab GPU is recommended for real runs
(see training/colab/README.md). Checkpoints are loadable by ml/runtime.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quasegpt import (  # noqa: E402
    QuaseGPT,
    QuaseGPTConfig,
    choose_device,
    load_checkpoint,
    save_checkpoint,
)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).long()
    return x.to(device), y.to(device)


# Architecture keys compared between a resume checkpoint's embedded config
# and the current training config. Weight-shape checks in load_checkpoint
# catch most mismatches, but keys like n_head / dropout / bias /
# tie_weights can differ while all tensor shapes stay identical.
ARCH_KEYS = (
    "model_type",
    "vocab_size",
    "block_size",
    "n_embd",
    "n_head",
    "n_layer",
    "dropout",
    "bias",
    "tie_weights",
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.json")
    ap.add_argument(
        "--resume",
        default=None,
        help="path to a checkpoint.pt saved by this training loop; "
        "resume from checkpoint step + 1 toward max_steps (final target)",
    )
    return ap.parse_args(argv)


def resume_training(model, opt, resume_path: str, device) -> int:
    """Load a training checkpoint into model + optimizer.

    Returns the global step to continue from (checkpoint step + 1).
    Raises a clear error if the checkpoint is missing, unrecognized,
    architecturally incompatible, or has no optimizer state.
    """
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
    obj = torch.load(resume_path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError(
            f"incompatible checkpoint {resume_path!r}: unrecognized format, "
            "expected {'model_state_dict': ..., 'optimizer_state_dict': ..., "
            "'step': ..., 'config': {...}}"
        )
    ckpt_cfg = obj.get("config")
    if isinstance(ckpt_cfg, dict):
        mismatches = [
            f"{k} (checkpoint={ckpt_cfg[k]!r}, config={getattr(model.config, k)!r})"
            for k in ARCH_KEYS
            if k in ckpt_cfg and ckpt_cfg[k] != getattr(model.config, k, None)
        ]
        if mismatches:
            raise ValueError(
                f"incompatible checkpoint {resume_path!r}: architecture mismatch: "
                + "; ".join(mismatches)
            )
    try:
        info = load_checkpoint(model, resume_path)  # strict: never hides mismatches
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"incompatible checkpoint {resume_path!r}: {e}") from e
    opt_state = obj.get("optimizer_state_dict")
    if opt_state is None:
        raise ValueError(
            f"incompatible checkpoint {resume_path!r}: no optimizer_state_dict; "
            "cannot resume optimizer state (was it saved without an optimizer?)"
        )
    opt.load_state_dict(opt_state)
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return int(info["step"]) + 1


def validate_start_step(start_step: int, max_steps: int, resume_path: str) -> None:
    if start_step >= max_steps:
        raise ValueError(
            f"checkpoint {resume_path!r} is already at step {start_step - 1} "
            f"but max_steps={max_steps}: nothing to train. "
            "Increase max_steps to continue training."
        )


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        tcfg = json.load(f)

    model_cfg = QuaseGPTConfig.from_dict(tcfg["model"])
    data_dir = tcfg.get("data_dir", "./data")
    out_dir = tcfg.get("out_dir", "./model")
    batch_size = tcfg.get("batch_size", 32)
    max_steps = tcfg.get("max_steps", 5000)
    lr = tcfg.get("learning_rate", 3e-4)
    weight_decay = tcfg.get("weight_decay", 0.1)
    warmup = tcfg.get("warmup_steps", 200)
    eval_every = tcfg.get("eval_every", 500)
    save_every = tcfg.get("save_every", 1000)
    seed = tcfg.get("seed", 42)

    torch.manual_seed(seed)
    device = choose_device(tcfg.get("device", "auto"))
    print(f"training on {device}")

    train = torch.load(os.path.join(data_dir, "train.pt")).long()
    val = torch.load(os.path.join(data_dir, "val.pt")).long()
    print(f"train tokens: {len(train)}, val tokens: {len(val)}")

    model = QuaseGPT(model_cfg).to(device)
    print(f"params: {model.num_parameters() / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))

    start_step = 0
    if args.resume:
        start_step = resume_training(model, opt, args.resume, device)
        validate_start_step(start_step, max_steps, args.resume)
        print(f"resuming checkpoint: {args.resume}")
        print(f"resuming from step: {start_step}")

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, max_steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    os.makedirs(out_dir, exist_ok=True)
    model.train()
    t0 = time.time()
    # NOTE: max_steps is the FINAL target step, not an additional count.
    # After --resume from step N, this runs steps N..max_steps-1, and the
    # LR schedule lr_at(step) always uses the real global step.
    for step in range(start_step, max_steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        x, y = get_batch(train, batch_size, model_cfg.block_size, device)
        logits, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step}/{max_steps} loss={loss.item():.4f} lr={lr_at(step):.2e} "
                  f"({time.time() - t0:.0f}s)")
        if step % eval_every == 0 or step == max_steps - 1:
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(val, batch_size, model_cfg.block_size, device)
                _, vloss = model(vx, vy)
            print(f"  val loss={vloss.item():.4f}")
            model.train()
        if step % save_every == 0 or step == max_steps - 1:
            model_cfg.to_json(os.path.join(out_dir, "config.json"))
            save_checkpoint(model, os.path.join(out_dir, "checkpoint.pt"), opt, step)
            print(f"  saved checkpoint @ step {step}")
    print("done.")


if __name__ == "__main__":
    main()
