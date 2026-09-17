"""Device selection: CUDA > MPS > CPU. Never crashes for missing CUDA."""
from __future__ import annotations

import os

import torch


def choose_device(preferred: str | None = None) -> torch.device:
    want = (preferred or os.environ.get("QUASEGPT_DEVICE") or "auto").lower()
    if want in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps":
        return torch.device("mps") if _mps_ok() else torch.device("cpu")
    if want == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_ok():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_ok() -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False
