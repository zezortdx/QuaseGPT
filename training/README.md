# Training

Colab is **optional** and **only for GPU-heavy training**. Inference never
needs it.

## Recommended flow

1. Work in this repo. The architecture in `ml/quasegpt/` is shared —
   Colab uses the same files, so there is never a "notebook fork" of the model.
2. In Colab: clone the repo, install `ml/requirements.txt`, upload or
   download your dataset (see Dataset below), run `python ml/train.py`.
3. Export artifacts (`training/colab/README.md`), download
   `quasegpt-export.zip`, unzip into local `./model/`.
4. `python scripts/verify_model.py`, restart the inference server.

## Local training

Possible but slow (CPU/MPS):

```
pip install -r ml/requirements.txt
python ml/prepare_data.py --input data/corpus.txt --out-dir data --train-tokenizer
python ml/train.py --config configs/training.json
```

## Dataset

`ml/prepare_data.py` accepts a `.txt` file or directory of `.txt` files.
TinyStories (roneneldan/TinyStories on HuggingFace) is the usual starter
corpus for small GPTs: plain-text children's stories, MIT-licensed script,
~2B tokens total — use a small slice for v0.1. Document the exact source and
slice you trained on; do not claim training data you cannot verify.

Nothing under `data/` or `checkpoints/` is committed (see `.gitignore`).
