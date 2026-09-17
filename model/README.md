# model/

Runtime artifacts live here. Large weights are gitignored; config + docs are committed.

## Expected files

```
model/
├── checkpoint.pt    # trained weights (NOT committed, see below)
├── config.json      # architecture — committed, source of truth at runtime
├── tokenizer.json   # BPE vocab + merges — committed (small, ~tens of KB)
└── README.md
```

## Committed placeholder vs real export

The `config.json` / `tokenizer.json` committed here form a **consistent
placeholder pair** (2000-token BPE trained on a tiny built-in English word
list). They exist so the app boots, tests run, and the full pipeline is
exercisable without weights. They are NOT the trained model: there is no
`checkpoint.pt` until you export one. A real Colab export **replaces both
files** (vocab size will change, typically to 8000) — that is expected and
supported; the loader validates the pair on every startup.

## Where to put the checkpoint

1. Export from Colab (see `training/colab/README.md`), producing
   `quasegpt-export/{checkpoint.pt,config.json,tokenizer.json}`.
2. Copy them here:
   ```
   cp quasegpt-export/checkpoint.pt model/checkpoint.pt
   cp quasegpt-export/config.json  model/config.json
   cp quasegpt-export/tokenizer.json model/tokenizer.json
   ```
3. Verify: `python scripts/verify_model.py`
4. Restart the inference server.

## What config.json represents

Exact architecture of the checkpoint: `vocab_size`, `block_size` (context),
`n_embd`, `n_head`, `n_layer`, `dropout`, `bias`, `tie_weights`.
The loader refuses to start on shape mismatch instead of silently misbehaving.

## Tokenizer artifacts

`tokenizer.json` holds the full BPE state (vocab + ordered merges + special
tokens `<pad> <unk> <bos> <eos>`). No extra vocab/merge sidecar files are
needed. Same file in Colab and locally => same text maps to same IDs.

## Compatibility check

`python scripts/verify_model.py` loads config + tokenizer + checkpoint on CPU,
reports parameter count, and runs a 16-token smoke generation. Anything red
there will also fail in the server — fix it before starting Rails.
