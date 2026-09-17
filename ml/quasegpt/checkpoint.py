"""Checkpoint save/load with strict architecture validation.

Supported formats (torch.save output):
  1. plain state_dict: {name: tensor, ...}
  2. training dict: {"model_state_dict": ..., "optimizer_state_dict": ...,
                     "step": ..., "config": {...}, ...}

Never silently ignores missing/unexpected weights: reports them and raises.
"""
from __future__ import annotations

import torch


def save_checkpoint(model, path: str, optimizer=None, step: int = 0,
                    extra: dict | None = None) -> None:
    payload: dict = {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj  # plain state_dict
    raise ValueError(
        "unrecognized checkpoint format: expected torch.save(model.state_dict()) "
        "or {'model_state_dict': ...}"
    )


def load_checkpoint(model, path: str, map_location=None, strict: bool = True) -> dict:
    """Load weights into `model`. Returns info dict {step, missing, unexpected}."""
    obj = torch.load(path, map_location=map_location or "cpu", weights_only=True)
    state = _extract_state_dict(obj)
    step = obj.get("step", 0) if isinstance(obj, dict) else 0

    # Architecture cross-check against config (embedding shapes carry the truth)
    cfg = model.config
    wte = state.get("wte.weight")
    if wte is not None and tuple(wte.shape) != (cfg.vocab_size, cfg.n_embd):
        raise ValueError(
            f"checkpoint wte.weight shape {tuple(wte.shape)} != "
            f"config (vocab_size={cfg.vocab_size}, n_embd={cfg.n_embd}); "
            "checkpoint architecture does not match config.json"
        )
    wpe = state.get("wpe.weight")
    if wpe is not None and wpe.shape[1] != cfg.n_embd:
        raise ValueError(
            f"checkpoint wpe dim {wpe.shape[1]} != config n_embd={cfg.n_embd}"
        )
    if wpe is not None and wpe.shape[0] != cfg.block_size:
        raise ValueError(
            f"checkpoint block_size {wpe.shape[0]} != config block_size={cfg.block_size}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return {"step": step, "missing": sorted(missing), "unexpected": sorted(unexpected)}
