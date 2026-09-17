"""Byte-fallback BPE tokenizer (pure stdlib, no external deps).

Format (tokenizer.json):
{
  "version": 1,
  "pattern": "<regex>",
  "special_tokens": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
  "vocab": {"token_str": id, ...},
  "merges": ["a b", ...]              # ordered by rank
}

Byte mapping: raw UTF-8 bytes <-> latin-1 chars, so every byte is exactly one
unicode char. JSON stays portable and encode() is total over all text.

Reproducibility contract: same input text -> same token IDs, inside or
outside Colab, as long as tokenizer.json is the same file.
"""
from __future__ import annotations

import json
import re
from collections import Counter

DEFAULT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
# NOTE: stdlib `re` has no \p{...}; we translate the few classes we use.
STD_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"

DEFAULT_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _bytes_to_latin(b: bytes) -> str:
    return b.decode("latin-1")


def _latin_to_bytes(s: str) -> bytes:
    return s.encode("latin-1")


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]],
                 special_tokens: dict[str, int] | None = None,
                 pattern: str = STD_PATTERN):
        self.vocab: dict[str, int] = dict(vocab)
        self.merges: list[tuple[str, str]] = list(merges)
        self.rank: dict[tuple[str, str], int] = {p: i for i, p in enumerate(self.merges)}
        self.inv_vocab: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self.special_set = set(self.special_tokens)
        self.pattern = pattern
        self._split_re = re.compile(pattern)
        # longest-match splitter for special tokens
        if self.special_set:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted(self.special_set, key=len, reverse=True)) + ")"
            )
        else:
            self._special_re = None

    # -- construction -----------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int,
              special_tokens: list[str] | None = None,
              pattern: str = STD_PATTERN) -> "BPETokenizer":
        specials = list(special_tokens or DEFAULT_SPECIALS)
        if vocab_size < len(specials) + 256:
            raise ValueError("vocab_size too small for specials + 256 bytes")
        vocab: dict[str, int] = {}
        for i, s in enumerate(specials):
            vocab[s] = i
        base = len(specials)
        for b in range(256):
            vocab[_bytes_to_latin(bytes([b]))] = base + b

        # Pre-tokenize into byte-char words
        words: list[list[str]] = []
        for m in re.compile(pattern).finditer(text):
            chunk = m.group(0)
            if chunk in specials:
                continue
            words.append([_bytes_to_latin(bytes([b])) for b in chunk.encode("utf-8")])

        merges: list[tuple[str, str]] = []
        while len(vocab) < vocab_size:
            stats: Counter[tuple[str, str]] = Counter()
            for w in words:
                for a, b in zip(w, w[1:]):
                    stats[(a, b)] += 1
            if not stats:
                break
            best = max(stats, key=lambda p: (stats[p], p))
            if stats[best] < 2:
                break
            new_tok = best[0] + best[1]
            merges.append(best)
            vocab[new_tok] = len(vocab)
            # apply merge to all words
            merged_words = []
            for w in words:
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                merged_words.append(out)
            words = merged_words
        special_ids = {s: vocab[s] for s in specials}
        return cls(vocab, merges, special_ids, pattern)

    # -- core bpe ----------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        # token is a latin-1 string (one char per byte)
        parts = list(token)
        if len(parts) <= 1:
            return parts
        while True:
            pairs = [(a, b) for a, b in zip(parts, parts[1:])]
            ranks = [(self.rank.get(p, 1 << 30), p) for p in pairs]
            best_rank, best_pair = min(ranks, key=lambda x: x[0])
            if best_rank == (1 << 30):
                break
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    out.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
            if len(parts) == 1:
                break
        return parts

    # -- public api ---------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos and "<bos>" in self.special_tokens:
            ids.append(self.special_tokens["<bos>"])
        # split out special tokens first so they survive splitting
        chunks = [text]
        if self._special_re is not None:
            split = self._special_re.split(text)
            chunks = [c for c in split if c]
        unk = self.special_tokens.get("<unk>")
        for chunk in chunks:
            if chunk in self.special_set:
                ids.append(self.special_tokens[chunk])
                continue
            for m in self._split_re.finditer(chunk):
                piece = m.group(0)
                raw = _bytes_to_latin(piece.encode("utf-8"))
                for sub in self._bpe(raw):
                    i = self.vocab.get(sub, unk)
                    if i is None:
                        raise ValueError(f"no encoding for {sub!r} and no <unk>")
                    ids.append(i)
        if add_eos and "<eos>" in self.special_tokens:
            ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        toks = []
        for i in ids:
            t = self.inv_vocab.get(i)
            if t is None:
                raise ValueError(f"unknown token id {i}")
            if skip_special and t in self.special_set:
                continue
            toks.append(t)
        raw = "".join(toks)
        try:
            return _latin_to_bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            return raw

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": dict(self.special_tokens),
            "vocab": dict(self.vocab),
            "merges": [f"{a} {b}" for a, b in self.merges],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {str(k): int(v) for k, v in data["vocab"].items()}
        merges: list[tuple[str, str]] = []
        for item in data.get("merges", []):
            # tokens may contain spaces; split on LAST space is wrong too.
            # We stored "a b" with a,b being latin-1 strings that can contain
            # spaces (byte 0x20). Disambiguate: merge line = a + " " + b where
            # a+b must already be in vocab at load position. Try every split.
            best = None
            for k in range(1, len(item)):
                if item[k] != " ":
                    continue
                a, b = item[:k], item[k + 1:]
                if a + b in vocab:
                    best = (a, b)
                    break
            if best is None:  # fallback: first space
                a, b = item.split(" ", 1)
                best = (a, b)
            merges.append(best)
        specials = {str(k): int(v) for k, v in data.get("special_tokens", {}).items()}
        return cls(vocab, merges, specials, data.get("pattern", STD_PATTERN))


def load_tokenizer(path: str) -> BPETokenizer:
    return BPETokenizer.load(path)
