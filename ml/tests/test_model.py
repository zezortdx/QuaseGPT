import torch

from quasegpt import QuaseGPT


def test_config_roundtrip(tmp_path, tiny_config):
    p = str(tmp_path / "c.json")
    tiny_config.to_json(p)
    from quasegpt import QuaseGPTConfig
    c2 = QuaseGPTConfig.from_json(p)
    assert c2.vocab_size == tiny_config.vocab_size
    assert c2.block_size == tiny_config.block_size


def test_model_construction_and_param_count(tiny_model, tiny_config):
    n = tiny_model.num_parameters()
    assert n > 0
    # manual check: embeddings + blocks exist
    assert tiny_model.wte.weight.shape == (tiny_config.vocab_size, tiny_config.n_embd)
    assert len(tiny_model.blocks) == tiny_config.n_layer


def test_forward_shape_and_loss(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, loss = tiny_model(x, x)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert loss is not None and loss.item() > 0


def test_forward_rejects_overflow(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size + 1))
    try:
        tiny_model(x)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overflow context")


def test_weight_tying(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.wte.weight
