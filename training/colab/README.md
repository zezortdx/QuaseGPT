# Colab training workflow (GPU only — never needed for inference)

`ml/train.py` runs as a subprocess (`!python ...`), so nothing it creates
exists as a notebook variable afterward. Training already saves its own
checkpoint — the export step below only packages files, never re-saves
the model from notebook variables.

Full flow: clone → install → verify GPU → prepare dataset → optional
smoke test → full training (or `--resume` after preemption) → confirm
`model/checkpoint.pt` → package three files →
download `quasegpt-export.zip` → install locally →
`python scripts/verify_model.py` → restart `bin/dev`.

## 1. Setup cell

`cd` inside a `!...` shell command does not persist across Colab cells,
so clone and then change directory with the `%cd` magic:

```python
!git clone https://github.com/zezortdx/QuaseGPT.git quasegpt
%cd /content/quasegpt
!pip install -q -r ml/requirements.txt
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If `CUDA: False`, stop here: switch the Colab runtime to a GPU
(Runtime → Change runtime type → T4 or better) before continuing.

## 2. Dataset cell

`ml/prepare_data.py` has no built-in downloader. It reads a `.txt` file
(or a directory of `.txt` files) that **you** provide and writes
`train.pt` / `val.pt` (+ `meta.json`). There is no automatic TinyStories
download — if you want TinyStories
(`roneneldan/TinyStories` on HuggingFace), download a slice yourself and
save it as `data/corpus.txt` first (via the Colab file browser or a
`!curl`/`!python` download cell of your own).

Then:

```python
!python ml/prepare_data.py --input data/corpus.txt --out-dir data \
    --tokenizer model/tokenizer.json --train-tokenizer --vocab-size 2000
```

(Keep `--vocab-size` in sync with `model.vocab_size` in
`configs/training.json`; a real export replaces both files anyway.)

This writes `data/train.pt`, `data/val.pt`, `data/meta.json`, and a fresh
`model/tokenizer.json`. Commit the tokenizer back to the repo (it is
small, tens of KB).

### Tokenizer warning (`--train-tokenizer`)

`--train-tokenizer` creates a brand-new tokenizer, overwriting
`model/tokenizer.json`. That is fine — and intended — before the first
training run. But after a checkpoint exists, re-training the tokenizer
changes vocab/merges, which can make the existing checkpoint incompatible
(shape mismatch) or silently semantically wrong (same IDs, different
words). Rule: **checkpoint + config + tokenizer belong together.** If you
re-train the tokenizer, you must re-train (or at least re-export all
three files from the same run) before using the checkpoint.

## 3. Smoke-training cell (optional but recommended, ~minutes)

Before the full 5000-step run, verify CUDA, data loading,
tokenizer/config match, forward + backward pass, finite loss, and
checkpoint saving — without touching `configs/training.json`:

```python
import json
with open("configs/training.json") as f:
    cfg = json.load(f)
cfg["max_steps"] = 200
cfg["eval_every"] = 50
cfg["save_every"] = 100
with open("/content/quasegpt-smoke.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(cfg)
```

```python
!python ml/train.py --config /content/quasegpt-smoke.json
```

Loss should generally trend downward. No specific final value is
required. If loss becomes `NaN`/`inf`, stays exactly constant, or
explodes, stop and debug (data, tokenizer/config mismatch, learning
rate) before starting the full run.

## 4. Train cell (full run)

```python
!python ml/train.py --config configs/training.json
```

What `ml/train.py` actually does (defaults from `configs/training.json`:
`out_dir=./model`, `max_steps=5000`, `save_every=1000`):

- Saves to `model/checkpoint.pt` (overwritten on every save, single file
  — there are no per-step checkpoint files).
- Saves on every `save_every` boundary (`step % save_every == 0`, i.e.
  steps 0, 1000, 2000, 3000, 4000) **and always at the final step**
  (`step == max_steps - 1`, i.e. step 4999).
- Writes `model/config.json` alongside every checkpoint save
  (`model_cfg.to_json(...)` on each save).
- Format (`ml/quasegpt/checkpoint.py:save_checkpoint`): a
  `torch.save` dict with `model_state_dict`, `config` (dict),
  `step`, and `optimizer_state_dict`. The local loader
  (`ml/runtime.py` / `scripts/verify_model.py`) reads exactly this.

Tune `configs/training.json` (`max_steps`, `batch_size`,
`learning_rate`) as needed.

## 5. Recovery (resume after preemption)

If Colab disconnects, don't restart from scratch — resume from the last
saved checkpoint. A Colab VM reset wipes local files, so during long runs
periodically copy the checkpoint somewhere persistent (Drive mount or a
manual download):

```python
# from a Drive-mounted notebook, e.g. after every few thousand steps:
!cp model/checkpoint.pt /content/gdrive/MyDrive/QuaseGPT-39M/checkpoint.pt
```

Then resume with the same config (or a copy with a larger `max_steps`):

```python
!python ml/train.py \
  --config /content/quasegpt-39m.json \
  --resume /content/gdrive/MyDrive/QuaseGPT-39M/checkpoint.pt
```

Semantics: the checkpoint's model + optimizer state are restored
(strictly validated against the config — an incompatible checkpoint is a
clear error, never a silent mismatch), training continues from
checkpoint step + 1 (e.g. a checkpoint saved at step 7500 resumes at
step 7501), and `max_steps` stays the FINAL target step (with
`max_steps=20000` it runs 7501..19999, not a new 20000-step run). The LR
schedule follows the real global step, and checkpoint saves continue
normally. Startup prints `resuming checkpoint: ...` and
`resuming from step: ...`.

## 6. Confirm + export cell (the important one)

Do NOT re-save the model from notebook variables (`model.state_dict()`,
`optimizer.state_dict()`, `step`, `config` do not exist after
`!python ml/train.py ...` — that ran in a separate process). Just verify
the files the training script already wrote, then package them:

```python
import os
required = [
    "model/checkpoint.pt",
    "model/config.json",
    "model/tokenizer.json",
]
for path in required:
    print(path, "OK" if os.path.exists(path) else "MISSING")
assert all(os.path.exists(path) for path in required), \
    "Missing model artifact. Make sure training completed successfully."
```

```python
import os
import shutil
os.makedirs("quasegpt-export", exist_ok=True)
shutil.copy("model/checkpoint.pt", "quasegpt-export/checkpoint.pt")
shutil.copy("model/config.json", "quasegpt-export/config.json")
shutil.copy("model/tokenizer.json", "quasegpt-export/tokenizer.json")
```

```python
!zip -j quasegpt-export.zip \
    quasegpt-export/checkpoint.pt \
    quasegpt-export/config.json \
    quasegpt-export/tokenizer.json
```

```python
from google.colab import files
files.download("quasegpt-export.zip")
```

Or run `python scripts/export_from_colab.py` to print these cells.

## 7. Download + install locally

1. Download `quasegpt-export.zip` (the `files.download` call above).
2. `unzip quasegpt-export.zip -d quasegpt-export`
3. `cp quasegpt-export/checkpoint.pt model/`
   `cp quasegpt-export/config.json model/`
   `cp quasegpt-export/tokenizer.json model/`
4. `python scripts/verify_model.py`
5. Restart `bin/dev` (required — the Rails app and the inference server
   read model artifacts at startup, so replacing files needs a restart).
