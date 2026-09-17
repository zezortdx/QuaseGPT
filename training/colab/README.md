# Colab training workflow (GPU only — never needed for inference)

## 1. Setup cell

```python
!git clone <YOUR-REPO-URL> quasegpt && cd quasegpt
!pip install -q -r ml/requirements.txt
import sys; sys.path.insert(0, "ml")
import torch; print(torch.cuda.is_available())
```

## 2. Dataset cell

Upload a `.txt` corpus (or download TinyStories slice) to `data/`, then:

```python
!python ml/prepare_data.py --input data/corpus.txt --out-dir data \
    --tokenizer model/tokenizer.json --train-tokenizer --vocab-size 2000
```
(Keep `--vocab-size` in sync with `model.vocab_size` in
`configs/training.json`; a real export replaces both files anyway.)

This writes `data/train.pt`, `data/val.pt`, and a fresh `model/tokenizer.json`.
Commit the tokenizer back to the repo (it is small).

## 3. Train cell

```python
!python ml/train.py --config configs/training.json
```

Tune `configs/training.json` (`max_steps`, `batch_size`, `learning_rate`).
Checkpoints land in `./model/checkpoint.pt` on the Colab VM.

## 4. Export cell (the important one)

```python
import torch, os
os.makedirs("quasegpt-export", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": step,
    "config": config.to_dict() if hasattr(config, "to_dict") else config,
}, "quasegpt-export/checkpoint.pt")
!cp model/config.json model/tokenizer.json quasegpt-export/
!cd quasegpt-export && zip ../quasegpt-export.zip checkpoint.pt config.json tokenizer.json
```

Or run `python scripts/export_from_colab.py` to print these cells.

## 5. Download + install locally

1. Download `quasegpt-export.zip` from the Colab file browser.
2. `unzip quasegpt-export.zip -d quasegpt-export`
3. `cp quasegpt-export/{checkpoint.pt,config.json,tokenizer.json} model/`
4. `python scripts/verify_model.py`
5. Restart: `bin/dev`

## Variable names

The export cell assumes the classic names `model`, `optimizer`, `step`,
`config`, `tokenizer`. If your notebook used others, substitute them —
the format (`model_state_dict` + `config` + `tokenizer.json`) is what the
local loader requires.
