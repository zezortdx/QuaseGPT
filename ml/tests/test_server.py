"""Server endpoint tests (no heavyweight checkpoint needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _client_with_tiny(tiny_model, tiny_tokenizer):
    import server
    from types import SimpleNamespace
    rt = SimpleNamespace(model=tiny_model, tokenizer=tiny_tokenizer,
                         config=tiny_model.config, device="cpu",
                         checkpoint_path=None, param_count=1234)
    server._state["runtime"] = rt
    server._state["error"] = None
    return TestClient(server.app)


def test_health(tiny_model, tiny_tokenizer):
    c = _client_with_tiny(tiny_model, tiny_tokenizer)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_generate(tiny_model, tiny_tokenizer):
    c = _client_with_tiny(tiny_model, tiny_tokenizer)
    r = c.post("/generate", json={"prompt": "hello world", "max_new_tokens": 8,
                                  "temperature": 0.0, "top_k": None})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["generated_tokens"] == 8
    assert isinstance(body["text"], str)


def test_generate_rejects_bad_params(tiny_model, tiny_tokenizer):
    c = _client_with_tiny(tiny_model, tiny_tokenizer)
    assert c.post("/generate", json={"prompt": "", "max_new_tokens": 8}).status_code == 422
    assert c.post("/generate", json={"prompt": "hi", "max_new_tokens": 99999}).status_code in (422, 500)


def test_stream(tiny_model, tiny_tokenizer):
    c = _client_with_tiny(tiny_model, tiny_tokenizer)
    r = c.post("/generate/stream", json={"prompt": "hello", "max_new_tokens": 4,
                                         "temperature": 0.0, "top_k": None})
    assert r.status_code == 200
    assert "done" in r.text


def test_missing_model_is_503():
    import server
    server._state["runtime"] = None
    server._state["error"] = "simulated missing checkpoint"
    c = TestClient(server.app, raise_server_exceptions=False)
    r = c.post("/generate", json={"prompt": "hi", "max_new_tokens": 4})
    assert r.status_code == 503
