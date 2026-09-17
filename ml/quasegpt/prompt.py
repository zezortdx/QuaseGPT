"""Chat adapter for a BASE language model (not instruction-tuned).

QuaseGPT v0.1 is a text-continuation model. The UI collects chat messages,
but the model only completes text. This module isolates that formatting so
the chat template can change without touching Rails controllers.

Base-model behavior may differ from an instruction-tuned chatbot:
it continues text plausibly rather than "following instructions".
"""
from __future__ import annotations


class PromptFormatter:
    def format(self, messages: list[dict]) -> str:
        """messages: [{role: user|assistant, content: str}] -> continuation prompt."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        parts.append("Assistant:")
        return "\n".join(parts) + " "


def build_context(messages: list[dict], tokenizer, block_size: int,
                  max_new_tokens: int, reserve: int = 16) -> str:
    """Format messages, newest-first truncation to fit block_size budget."""
    fmt = PromptFormatter()
    budget = max(16, block_size - max_new_tokens - reserve)
    # Keep newest messages that fit; always keep the last user message.
    kept: list[dict] = []
    for m in reversed(messages):
        trial = list(reversed(kept + [m]))
        ids = tokenizer.encode(fmt.format([{"role": x["role"], "content": x["content"]} for x in trial]))
        if len(ids) <= budget or not kept:
            kept.append(m)
        else:
            break
    ordered = list(reversed(kept))
    text = fmt.format([{"role": m["role"], "content": m["content"]} for m in ordered])
    ids = tokenizer.encode(text)
    if len(ids) > budget:
        ids = ids[-budget:]
        text = tokenizer.decode(ids)
        if "Assistant:" not in text:
            text += "\nAssistant: "
    return text


def conversation_title(first_user_message: str, limit: int = 50) -> str:
    """Deterministic title: trimmed first user message. No LLM involved."""
    t = " ".join((first_user_message or "").split())
    if len(t) <= limit:
        return t or "New chat"
    cut = t[:limit].rsplit(" ", 1)[0] or t[:limit]
    return cut + "…"
