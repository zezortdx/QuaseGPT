"""Shared pytest fixtures. TEST-ONLY tiny model — never used in production."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quasegpt"))

from quasegpt import BPETokenizer, QuaseGPT, QuaseGPTConfig  # noqa: E402

TINY = QuaseGPTConfig(model_type="quasegpt", vocab_size=320, block_size=32,
                      n_embd=48, n_head=4, n_layer=2, dropout=0.0)


@pytest.fixture()
def tiny_config():
    return TINY


@pytest.fixture()
def tiny_tokenizer(tmp_path):
    text = ("hello world this is a tiny test corpus for quasegpt. " * 40
            + "the quick brown fox jumps over the lazy dog. " * 40)
    tok = BPETokenizer.train(text, 320)
    p = str(tmp_path / "tok.json")
    tok.save(p)
    return BPETokenizer.load(p)


@pytest.fixture()
def tiny_model(tiny_config):
    torch.manual_seed(0)
    m = QuaseGPT(tiny_config)
    m.eval()
    return m


@pytest.fixture()
def tiny_checkpoint(tmp_path, tiny_model):
    from quasegpt import save_checkpoint
    p = str(tmp_path / "ckpt.pt")
    save_checkpoint(tiny_model, p, step=7)
    return p
