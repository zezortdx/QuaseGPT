"""Shared runtime loader: the ONE path used by chat CLI and the server.

No duplicate inference implementation: server.py and chat.py both load via
load_runtime() from this module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from quasegpt import (
    BPETokenizer,
    QuaseGPT,
    QuaseGPTConfig,
    choose_device,
    load_checkpoint,
)


@dataclass
class Runtime:
    model: QuaseGPT
    tokenizer: BPETokenizer
    config: QuaseGPTConfig
    device: torch.device
    checkpoint_path: str | None
    param_count: int


def model_dir() -> str:
    return os.environ.get("QUASEGPT_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))


def load_runtime(model_dir_path: str | None = None, device: str | None = None) -> Runtime:
    mdir = model_dir_path or os.environ.get("QUASEGPT_MODEL_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "model")
    mdir = os.path.abspath(mdir)
    config_path = os.path.join(mdir, "config.json")
    tok_path = os.path.join(mdir, "tokenizer.json")
    ckpt_pt = os.path.join(mdir, "checkpoint.pt")
    ckpt = ckpt_pt if os.path.exists(ckpt_pt) else None

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model config missing: {config_path}")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer missing: {tok_path}")
    config = QuaseGPTConfig.from_json(config_path)
    tokenizer = BPETokenizer.load(tok_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} != config vocab_size {config.vocab_size}"
        )
    dev = choose_device(device)
    model = QuaseGPT(config).to(dev)
    loaded = None
    if ckpt is not None:
        info = load_checkpoint(model, ckpt, map_location=dev)
        loaded = ckpt
    model.eval()
    n = model.num_parameters()
    print("QuaseGPT inference")
    print("------------------")
    print(f"Model: QuaseGPT ({config.n_layer}L/{config.n_head}H/d{config.n_embd})")
    print(f"Parameters: {n / 1e6:.1f}M")
    print(f"Context: {config.block_size}")
    print(f"Device: {dev}")
    print(f"Checkpoint: {'loaded from ' + loaded if loaded else 'MISSING (random weights!)'}")
    print(f"Tokenizer: loaded ({tokenizer.vocab_size} tokens)")
    return Runtime(model, tokenizer, config, dev, loaded, n)
