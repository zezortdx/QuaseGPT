"""Explicit model configuration. No hidden architectural constants elsewhere."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class QuaseGPTConfig:
    """Canonical v0.1 architecture defaults.

    NOTE: no Colab notebook was present in the workspace at migration time
    (searched for *.ipynb, QuaseGPT.ipynb, training.ipynb — none found), so
    these are declared as the canonical v0.1 defaults, not recovered values.
    If you export from a real Colab run, config.json produced by
    scripts/export_from_colab.py is the source of truth for that checkpoint.
    """

    model_type: str = "quasegpt"
    vocab_size: int = 8000
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True

    def __post_init__(self) -> None:
        assert self.vocab_size >= 260, "vocab_size must leave room for specials + 256 bytes"
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size >= 8, "block_size too small"
        assert 0.0 <= self.dropout < 1.0, "dropout out of range"
        assert self.n_layer >= 1 and self.n_head >= 1

    @classmethod
    def from_json(cls, path: str) -> "QuaseGPTConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_dict(cls, data: dict) -> "QuaseGPTConfig":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
