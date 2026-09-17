"""Dataset preparation: tokenize raw text into train/val .pt shards.

Usage:
  python ml/prepare_data.py --input data/tinystories.txt --out-dir data \\
      --tokenizer model/tokenizer.json --block-size 256

Input may be a .txt file (one document per line, blank line = doc boundary
also accepted) or a directory of .txt files. Output: train.pt / val.pt with
int32 token streams + meta.json. Large datasets are NOT committed (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quasegpt"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quasegpt import BPETokenizer  # noqa: E402


def read_documents(path: str) -> list[str]:
    if os.path.isdir(path):
        docs = []
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                if fn.endswith(".txt"):
                    with open(os.path.join(root, fn), encoding="utf-8") as f:
                        docs.append(f.read())
        return docs
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # blank-line separated documents, fallback to lines
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    return docs or [text]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--tokenizer", default="./model/tokenizer.json")
    ap.add_argument("--train-tokenizer", action="store_true",
                    help="train a new BPE tokenizer on --input instead of loading")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    docs = read_documents(args.input)
    print(f"documents: {len(docs)}")

    if args.train_tokenizer:
        tok = BPETokenizer.train("\n".join(docs[:10000]), args.vocab_size)
        os.makedirs(os.path.dirname(os.path.abspath(args.tokenizer)), exist_ok=True)
        tok.save(args.tokenizer)
        print(f"trained tokenizer -> {args.tokenizer} ({tok.vocab_size} tokens)")
    else:
        tok = BPETokenizer.load(args.tokenizer)

    eos = tok.special_tokens.get("<eos>")
    stream: list[int] = []
    for d in docs:
        stream.extend(tok.encode(d))
        if eos is not None:
            stream.append(eos)
    print(f"total tokens: {len(stream)}")
    n_val = int(len(stream) * args.val_fraction)
    val = stream[:n_val]
    train = stream[n_val:]
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(torch.tensor(train, dtype=torch.int32), os.path.join(args.out_dir, "train.pt"))
    torch.save(torch.tensor(val, dtype=torch.int32), os.path.join(args.out_dir, "val.pt"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"train_tokens": len(train), "val_tokens": len(val),
                   "vocab_size": tok.vocab_size}, f, indent=2)
    print(f"wrote {args.out_dir}/train.pt ({len(train)}) + val.pt ({len(val)})")


if __name__ == "__main__":
    main()
