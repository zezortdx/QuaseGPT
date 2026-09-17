"""Standalone sanity-check CLI. Uses the SAME loader as the Rails path.

Usage: python ml/chat.py [--model-dir ./model] [--temperature 0.8]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from runtime import load_runtime  # noqa: E402
from quasegpt import generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="QuaseGPT local chat (no Colab needed)")
    ap.add_argument("--model-dir", default=os.environ.get("QUASEGPT_MODEL_DIR", "./model"))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    try:
        rt = load_runtime(args.model_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}")
        print("Export artifacts from Colab first — see model/README.md.")
        sys.exit(1)

    print(f"\nQuaseGPT v0.1  (running on {rt.device}, {rt.param_count / 1e6:.1f}M params)")
    print("Type a prompt. Empty line quits.\n")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        with torch.inference_mode():
            try:
                text, n = generate(rt.model, rt.tokenizer, prompt,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_k=args.top_k)
            except ValueError as e:
                print(f"quasegpt> [error] {e}")
                continue
        print(f"quasegpt> {text}\n")


if __name__ == "__main__":
    main()
