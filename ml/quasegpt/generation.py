"""Autoregressive generation: sampling, top-k/top-p, streaming."""
from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F

MAX_NEW_TOKENS_LIMIT = 1024
MAX_PROMPT_TOKENS = 8192


def validate_params(prompt: str, max_new_tokens: int, temperature: float,
                    top_k: int | None, top_p: float | None) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > 32_000:
        raise ValueError("prompt too long (max 32000 chars)")
    if not (1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT):
        raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS_LIMIT}")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be 0.0..2.0")
    if top_k is not None and not (1 <= top_k <= 1000):
        raise ValueError("top_k must be 1..1000")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be 0 < top_p <= 1.0")


def _sample(logits: torch.Tensor, temperature: float,
            top_k: int | None, top_p: float | None,
            generator: torch.Generator | None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum - probs > top_p
        sorted_logits[mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    if generator is not None:
        return torch.multinomial(probs, num_samples=1, generator=generator)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: int | None = 40,
             top_p: float | None = None, seed: int | None = None,
             stop_at_eos: bool = True) -> tuple[str, int]:
    """Full generation. Returns (text, generated_token_count)."""
    validate_params(prompt, max_new_tokens, temperature, top_k, top_p)
    device = next(model.parameters()).device
    block_size = model.config.block_size
    ids = tokenizer.encode(prompt)
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt encodes to {len(ids)} tokens (max {MAX_PROMPT_TOKENS})")
    ids = ids[-block_size:]  # crop to context
    x = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
        torch.manual_seed(seed)
    eos = tokenizer.special_tokens.get("<eos>")
    produced = 0
    model.eval()
    for _ in range(max_new_tokens):
        xc = x[:, -block_size:]
        logits, _ = model(xc)
        nxt = _sample(logits[:, -1, :], temperature, top_k, top_p, gen if seed is not None else None)
        x = torch.cat([x, nxt.to(x.device)], dim=1)
        produced += 1
        if stop_at_eos and eos is not None and int(nxt.item()) == eos:
            break
    new_ids = x[0, len(ids):].tolist()
    return tokenizer.decode(new_ids), produced


@torch.inference_mode()
def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int = 128,
                    temperature: float = 0.8, top_k: int | None = 40,
                    top_p: float | None = None, seed: int | None = None,
                    stop_at_eos: bool = True) -> Iterator[str]:
    """Yield incremental decoded deltas (decode-so-far diffed per step)."""
    validate_params(prompt, max_new_tokens, temperature, top_k, top_p)
    device = next(model.parameters()).device
    block_size = model.config.block_size
    ids = tokenizer.encode(prompt)[-block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    eos = tokenizer.special_tokens.get("<eos>")
    model.eval()
    new_ids: list[int] = []
    prev_text = ""
    for _ in range(max_new_tokens):
        xc = x[:, -block_size:]
        logits, _ = model(xc)
        nxt = _sample(logits[:, -1, :], temperature, top_k, top_p, gen if seed is not None else None)
        nid = int(nxt.item())
        x = torch.cat([x, nxt.to(x.device)], dim=1)
        if stop_at_eos and eos is not None and nid == eos:
            break
        new_ids.append(nid)
        text = tokenizer.decode(new_ids)
        delta = text[len(prev_text):]
        prev_text = text
        if delta:
            yield delta
