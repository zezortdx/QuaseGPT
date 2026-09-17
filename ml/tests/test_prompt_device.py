import torch

from quasegpt import build_context, conversation_title
from quasegpt.prompt import PromptFormatter


def test_prompt_format(tiny_tokenizer):
    f = PromptFormatter()
    p = f.format([{"role": "user", "content": "Tell me a story"}])
    assert "Tell me a story" in p and p.rstrip().endswith("Assistant:")


def test_context_truncation(tiny_tokenizer):
    msgs = [{"role": "user", "content": f"message number {i} " * 10} for i in range(10)]
    ctx = build_context(msgs, tiny_tokenizer, block_size=128, max_new_tokens=8)
    assert len(tiny_tokenizer.encode(ctx)) <= 128
    assert "message number 9" in ctx  # newest retained (tail-crop keeps the end)


def test_title():
    assert conversation_title("Tell me a story about dragons") == "Tell me a story about dragons"
    long = "x " * 100
    assert len(conversation_title(long)) <= 55
    assert conversation_title("") == "New chat"


def test_device_fallback(monkeypatch):
    from quasegpt import choose_device
    monkeypatch.setenv("QUASEGPT_DEVICE", "cpu")
    assert str(choose_device()) == "cpu"
    monkeypatch.delenv("QUASEGPT_DEVICE", raising=False)
    d = choose_device("cpu")
    assert str(d) == "cpu"
    # unknown cuda box must not crash
    assert choose_device("cpu") is not None
    assert torch.device("cpu") is not None
