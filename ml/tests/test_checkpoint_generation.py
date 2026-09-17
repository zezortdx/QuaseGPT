import pytest
import torch

from quasegpt import generate, generate_stream, load_checkpoint


def test_checkpoint_save_load(tiny_model, tiny_checkpoint):
    from quasegpt import QuaseGPT
    m2 = QuaseGPT(tiny_model.config)
    info = load_checkpoint(m2, tiny_checkpoint)
    assert info["step"] == 7
    for a, b in zip(tiny_model.parameters(), m2.parameters()):
        assert torch.allclose(a, b)


def test_checkpoint_mismatch_reported(tiny_model, tmp_path):
    from quasegpt import QuaseGPT, QuaseGPTConfig, save_checkpoint
    bad_cfg = QuaseGPTConfig(vocab_size=320, block_size=32, n_embd=64,
                             n_head=4, n_layer=2, dropout=0.0)
    bad = QuaseGPT(bad_cfg)
    p = str(tmp_path / "bad.pt")
    save_checkpoint(bad, p)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(tiny_model, p)


def test_generation_shape(tiny_model, tiny_tokenizer):
    text, n = generate(tiny_model, tiny_tokenizer, "hello world",
                       max_new_tokens=8, temperature=0.0, top_k=None)
    assert n == 8
    assert isinstance(text, str)


def test_generation_crops_long_context(tiny_model, tiny_tokenizer):
    long_prompt = "hello world this is a test " * 50
    text, n = generate(tiny_model, tiny_tokenizer, long_prompt,
                       max_new_tokens=4, temperature=0.0, top_k=None)
    assert n == 4 and isinstance(text, str)


def test_param_validation(tiny_model, tiny_tokenizer):
    with pytest.raises(ValueError):
        generate(tiny_model, tiny_tokenizer, "", max_new_tokens=8)
    with pytest.raises(ValueError):
        generate(tiny_model, tiny_tokenizer, "hi", max_new_tokens=99999)
    with pytest.raises(ValueError):
        generate(tiny_model, tiny_tokenizer, "hi", temperature=99)


def test_stream_yields_deltas(tiny_model, tiny_tokenizer):
    deltas = list(generate_stream(tiny_model, tiny_tokenizer, "hello",
                                  max_new_tokens=6, temperature=0.0, top_k=None))
    assert 0 < len(deltas) <= 6
    full, _ = generate(tiny_model, tiny_tokenizer, "hello",
                       max_new_tokens=6, temperature=0.0, top_k=None)
    assert "".join(deltas) == full
