"""Resume tests for ml/train.py --resume. CPU-only, tiny model, no real training."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import train  # noqa: E402
from quasegpt import QuaseGPT, QuaseGPTConfig, save_checkpoint  # noqa: E402

DEVICE = torch.device("cpu")


def _trained_state(tiny_model, steps=1):
    """Run a few optimizer steps so Adam state exists and weights move."""
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-4)
    tiny_model.train()
    for _ in range(steps):
        x = torch.randint(0, tiny_model.config.vocab_size,
                          (2, tiny_model.config.block_size))
        _, loss = tiny_model(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return opt


def test_fresh_defaults_unchanged():
    args = train.parse_args([])
    assert args.resume is None
    assert args.config == "configs/training.json"
    # fresh training starts at step 0 and passes validation
    train.validate_start_step(0, 5000, "unused.pt")


def test_resume_restores_model(tmp_path, tiny_model):
    opt = _trained_state(tiny_model)
    p = str(tmp_path / "ckpt.pt")
    save_checkpoint(tiny_model, p, opt, step=5)
    before = [t.detach().clone() for t in tiny_model.parameters()]

    fresh = QuaseGPT(tiny_model.config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    assert not all(torch.allclose(a, b)
                   for a, b in zip(before, fresh.parameters()))
    start = train.resume_training(fresh, fresh_opt, p, DEVICE)
    assert start == 6
    for a, b in zip(before, fresh.parameters()):
        assert torch.allclose(a, b)


def test_resume_restores_optimizer(tmp_path, tiny_model):
    opt = _trained_state(tiny_model, steps=2)
    p = str(tmp_path / "ckpt.pt")
    save_checkpoint(tiny_model, p, opt, step=5)
    saved_state = opt.state_dict()["state"]

    fresh = QuaseGPT(tiny_model.config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    train.resume_training(fresh, fresh_opt, p, DEVICE)
    restored = fresh_opt.state_dict()["state"]
    assert set(restored) == set(saved_state)
    for k in saved_state:
        for sk, sv in saved_state[k].items():
            rv = restored[k][sk]
            if isinstance(sv, torch.Tensor):
                assert torch.allclose(rv.cpu(), sv.cpu())
            else:
                assert rv == sv


def test_step_resumes_at_checkpoint_plus_one(tmp_path, tiny_model):
    opt = _trained_state(tiny_model)
    p = str(tmp_path / "ckpt.pt")
    save_checkpoint(tiny_model, p, opt, step=7500)
    fresh = QuaseGPT(tiny_model.config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    assert train.resume_training(fresh, fresh_opt, p, DEVICE) == 7501


def test_max_steps_remains_final_target():
    # 7500-step checkpoint + max_steps=20000 continues 7501..19999
    train.validate_start_step(7501, 20000, "ckpt.pt")
    assert list(range(7501, 20000))[-1] == 19999
    assert len(range(7501, 20000)) == 20000 - 7501
    # exhausted checkpoint is a clear error, not a silent restart
    with pytest.raises(ValueError, match="nothing to train"):
        train.validate_start_step(20000, 20000, "ckpt.pt")
    with pytest.raises(ValueError, match="nothing to train"):
        train.validate_start_step(20001, 20000, "ckpt.pt")


def test_incompatible_shape_fails_clearly(tmp_path, tiny_model):
    bad_cfg = QuaseGPTConfig(vocab_size=320, block_size=32, n_embd=64,
                             n_head=4, n_layer=2, dropout=0.0)
    bad = QuaseGPT(bad_cfg)
    bad_opt = torch.optim.AdamW(bad.parameters(), lr=3e-4)
    p = str(tmp_path / "bad.pt")
    save_checkpoint(bad, p, bad_opt, step=3)
    fresh = QuaseGPT(tiny_model.config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        train.resume_training(fresh, fresh_opt, p, DEVICE)


def test_incompatible_head_count_fails_clearly(tmp_path, tiny_model):
    # same tensor shapes (n_embd=48 divisible by 2 and 4) but different
    # architecture: must fail via config comparison, not load silently
    bad_cfg = QuaseGPTConfig(vocab_size=320, block_size=32, n_embd=48,
                             n_head=2, n_layer=2, dropout=0.0)
    bad = QuaseGPT(bad_cfg)
    bad_opt = torch.optim.AdamW(bad.parameters(), lr=3e-4)
    p = str(tmp_path / "bad-heads.pt")
    save_checkpoint(bad, p, bad_opt, step=3)
    fresh = QuaseGPT(tiny_model.config)  # n_head=4
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        train.resume_training(fresh, fresh_opt, p, DEVICE)


def test_missing_optimizer_state_fails_clearly(tmp_path, tiny_model):
    p = str(tmp_path / "noopt.pt")
    save_checkpoint(tiny_model, p, step=3)  # no optimizer
    fresh = QuaseGPT(tiny_model.config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    with pytest.raises(ValueError, match="no optimizer_state_dict"):
        train.resume_training(fresh, fresh_opt, p, DEVICE)


def test_missing_file_fails_clearly(tiny_model):
    fresh_opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-4)
    with pytest.raises(FileNotFoundError, match="not found"):
        train.resume_training(tiny_model, fresh_opt,
                              "/nonexistent/checkpoint.pt", DEVICE)
