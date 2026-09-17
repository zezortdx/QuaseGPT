"""Print the correct Colab setup/training/export cells.

`!python ml/train.py ...` runs in a separate process, so notebook
variables like `model` / `optimizer` / `step` / `config` do NOT exist
afterward. Training already saves `model/checkpoint.pt` itself — the
export flow below only verifies and packages files.

Run locally to inspect:

    python scripts/export_from_colab.py
"""
from __future__ import annotations

CELLS = '''# Cell 1 — setup (Colab): clone, enter dir (cd in !... does not persist), install, check GPU
!git clone https://github.com/zezortdx/QuaseGPT.git quasegpt
%cd /content/quasegpt
!pip install -q -r ml/requirements.txt
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Cell 2 — dataset: YOU provide data/corpus.txt (prepare_data.py has no downloader)
# (Optional) download a TinyStories slice yourself into data/corpus.txt first.
!python ml/prepare_data.py --input data/corpus.txt --out-dir data \\
    --tokenizer model/tokenizer.json --train-tokenizer --vocab-size 2000
# WARNING: --train-tokenizer overwrites model/tokenizer.json. Fine before the
# first training run; after a checkpoint exists, checkpoint + config +
# tokenizer belong together — re-training the tokenizer invalidates old weights.

# Cell 3 — optional smoke test (does NOT modify configs/training.json):
import json
with open("configs/training.json") as f:
    cfg = json.load(f)
cfg["max_steps"] = 200
cfg["eval_every"] = 50
cfg["save_every"] = 100
with open("/content/quasegpt-smoke.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(cfg)
!python ml/train.py --config /content/quasegpt-smoke.json
# Loss should generally trend downward (no specific target value).
# If loss is NaN/inf/constant/exploding, stop and debug before the full run.

# Cell 4 — full training (saves model/checkpoint.pt + model/config.json itself):
!python ml/train.py --config configs/training.json

# Cell 5 — verify the files training already wrote (no notebook variables used):
import os
required = [
    "model/checkpoint.pt",
    "model/config.json",
    "model/tokenizer.json",
]
for path in required:
    print(path, "OK" if os.path.exists(path) else "MISSING")
assert all(os.path.exists(path) for path in required), \\
    "Missing model artifact. Make sure training completed successfully."

# Cell 6 — package:
import os
import shutil
os.makedirs("quasegpt-export", exist_ok=True)
shutil.copy("model/checkpoint.pt", "quasegpt-export/checkpoint.pt")
shutil.copy("model/config.json", "quasegpt-export/config.json")
shutil.copy("model/tokenizer.json", "quasegpt-export/tokenizer.json")

# Cell 7 — zip + download:
!zip -j quasegpt-export.zip \\
    quasegpt-export/checkpoint.pt \\
    quasegpt-export/config.json \\
    quasegpt-export/tokenizer.json
from google.colab import files
files.download("quasegpt-export.zip")

# Locally after download:
#   unzip quasegpt-export.zip -d quasegpt-export
#   cp quasegpt-export/checkpoint.pt model/
#   cp quasegpt-export/config.json model/
#   cp quasegpt-export/tokenizer.json model/
#   python scripts/verify_model.py
#   (restart bin/dev — model artifacts are read at startup)
'''

if __name__ == "__main__":
    print(CELLS)
