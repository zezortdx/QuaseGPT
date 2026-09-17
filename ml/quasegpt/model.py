"""QuaseGPT decoder-only Transformer: embeddings + blocks + tied LM head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block
from .config import QuaseGPTConfig


class QuaseGPT(nn.Module):
    def __init__(self, config: QuaseGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # learned positions
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config.n_embd, config.n_head, config.block_size, config.dropout, config.bias)
             for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}; "
                "crop input before calling forward()"
            )
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def crop_block_size(self, new_block_size: int) -> None:
        """Reduce positional embeddings for a smaller context (loading compat)."""
        assert new_block_size <= self.config.block_size
        self.config.block_size = new_block_size
        self.wpe.weight = nn.Parameter(self.wpe.weight[:new_block_size])
        for block in self.blocks:
            if hasattr(block.attn, "mask"):
                block.attn.mask = torch.tril(
                    torch.ones(new_block_size, new_block_size)
                ).view(1, 1, new_block_size, new_block_size).to(block.attn.mask.device)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            n -= self.wpe.weight.numel()
        return n
