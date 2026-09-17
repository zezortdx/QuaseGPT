"""End-to-end smoke test: server health + generate (needs server running).

Usage:
  1. QUASEGPT_MODEL_DIR=./model python ml/server.py &   (or bin/dev)
  2. python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("QUASEGPT_INFERENCE_URL", "http://127.0.0.1:8000")


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.load(r)


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> int:
    try:
        h = _get("/health")
    except Exception as e:
        print(f"FAIL: server unreachable at {BASE}: {e}")
        return 1
    print(f"health: {h}")
    if not h.get("model_loaded"):
        print(f"FAIL: model not loaded: {h.get('error')}")
        return 1
    if h.get("checkpoint"):
        print(f"checkpoint: {h['checkpoint']}")
    else:
        print("NOTE: no checkpoint.pt exported yet — generating with random "
              "weights (plumbing check only). Export real weights per model/README.md.")
    g = _post("/generate", {"prompt": "Once upon a time", "max_new_tokens": 16,
                            "temperature": 0.8, "top_k": 40})
    print(f"generate: {g}")
    if not g.get("text"):
        print("FAIL: empty generation")
        return 1
    if h.get("checkpoint"):
        print("SMOKE OK — response came from the local QuaseGPT checkpoint.")
    else:
        print("SMOKE OK (plumbing only) — export checkpoint.pt for real output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
