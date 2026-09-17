"""Verify local artifacts: config + tokenizer + checkpoint load on CPU.

Usage: python scripts/verify_model.py [--model-dir ./model]
Exit 0 = ready. Nonzero = exact problem printed (no faking).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))

from quasegpt import (  # noqa: E402
    BPETokenizer,
    QuaseGPT,
    QuaseGPTConfig,
    generate,
    load_checkpoint,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./model")
    args = ap.parse_args()
    mdir = os.path.abspath(args.model_dir)
    cfg_p = os.path.join(mdir, "config.json")
    tok_p = os.path.join(mdir, "tokenizer.json")
    ckpt_p = os.path.join(mdir, "checkpoint.pt")

    print(f"model dir: {mdir}")
    if not os.path.exists(cfg_p):
        print(f"FAIL: missing {cfg_p}"); return 1
    if not os.path.exists(tok_p):
        print(f"FAIL: missing {tok_p}"); return 1
    if not os.path.exists(ckpt_p):
        print(f"FAIL: missing {ckpt_p} (export from Colab first — see model/README.md)")
        return 1

    config = QuaseGPTConfig.from_json(cfg_p)
    print(f"config: {config}")
    tok = BPETokenizer.load(tok_p)
    print(f"tokenizer: {tok.vocab_size} tokens, specials={tok.special_tokens}")
    if tok.vocab_size != config.vocab_size:
        print(f"FAIL: tokenizer vocab {tok.vocab_size} != config {config.vocab_size}")
        return 1

    s = "Once upon a time"
    ids = tok.encode(s)
    assert tok.decode(ids) == s, "tokenizer roundtrip broken"
    print(f"tokenizer roundtrip ok: {ids}")

    model = QuaseGPT(config)
    info = load_checkpoint(model, ckpt_p, map_location="cpu")
    print(f"checkpoint ok: step={info['step']} params={model.num_parameters() / 1e6:.2f}M")
    model.eval()
    with torch.inference_mode():
        text, n = generate(model, tok, "Once upon a time", max_new_tokens=16,
                           temperature=0.8, top_k=40)
    print(f"smoke generation ({n} tokens): {text!r}")
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
