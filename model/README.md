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

## Committed config/tokenizer vs weights

The `config.json` / `tokenizer.json` committed here describe the trained
QuaseGPT 39M (2000-token BPE trained on the training corpus; 12 layers /
8 heads / d512, context 256, ≈38.98M params at step 19999). They ship so
the app boots and tests run, but they are NOT the trained model on their
own: there is no `checkpoint.pt` in Git. A fresh Colab export **replaces
all three files as a set** — checkpoint + config + tokenizer always
belong together (see the tokenizer warning in `training/colab/README.md`).

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

> **Warning:** a checkpoint only works with the tokenizer it was trained
> with. Using a different `tokenizer.json` (different vocab size or
> merges) either fails loudly on shape mismatch or — worse — silently
> maps the same IDs to different words. Always keep the exported triple
> (`checkpoint.pt` + `config.json` + `tokenizer.json`) together.

## Compatibility check

`python scripts/verify_model.py` loads config + tokenizer + checkpoint on CPU,
reports parameter count, and runs a 16-token smoke generation. Anything red
there will also fail in the server — fix it before starting Rails.
