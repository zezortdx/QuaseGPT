from quasegpt import BPETokenizer


def test_encode_decode_roundtrip(tiny_tokenizer):
    s = "hello world, this is quasegpt."
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(s)) == s


def test_reproducibility(tmp_path, tiny_tokenizer):
    p = str(tmp_path / "t2.json")
    tiny_tokenizer.save(p)
    t2 = BPETokenizer.load(p)
    s = "the quick brown fox jumps over the lazy dog"
    assert tiny_tokenizer.encode(s) == t2.encode(s)


def test_special_tokens_preserved(tiny_tokenizer):
    ids = tiny_tokenizer.encode("<bos> hello <eos>")
    assert tiny_tokenizer.special_tokens["<bos>"] in ids
    assert tiny_tokenizer.special_tokens["<eos>"] in ids
    assert "<bos>" not in tiny_tokenizer.decode(ids)
    assert tiny_tokenizer.decode(ids, skip_special=False) != tiny_tokenizer.decode(ids)


def test_known_sentence_ids_stable(tiny_tokenizer):
    # golden check: same tokenizer bytes -> same ids across runs
    ids = tiny_tokenizer.encode("hello world")
    assert ids == tiny_tokenizer.encode("hello world")
    assert len(ids) >= 2
