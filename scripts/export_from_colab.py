"""Build the exact Colab export cells from THIS repo's modules.

Run inside Colab after training (or locally to inspect):

    python scripts/export_from_colab.py --help

It prints the notebook cells you need, tailored to variable names found in
the classic QuaseGPT-style notebook (model, optimizer, tokenizer, config).
If your notebook uses different names, the printed cell shows where to edit.
"""
from __future__ import annotations

CELLS = '''# Cell 1 — clone source (Colab): share ONE implementation, no forks
!git clone <YOUR-REPO-URL> quasegpt && cd quasegpt && pip install -q torch

# Cell 2 — after training, export artifacts (adjust variable names to yours):
import torch, json, os
os.makedirs("quasegpt-export", exist_ok=True)

# `model` = your trained nn.Module, `optimizer` = your AdamW, `step` = int
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict() if 'optimizer' in dir() else None,
    "step": step if 'step' in dir() else 0,
    "config": config if isinstance(config, dict) else config.to_dict(),
}, "quasegpt-export/checkpoint.pt")

# tokenizer: if you trained with ml/quasegpt/tokenizer.py in Colab:
from ml.quasegpt import BPETokenizer  # via sys.path to the cloned repo
tokenizer.save("quasegpt-export/tokenizer.json")

# config (if it was a QuaseGPTConfig object):
config.to_json("quasegpt-export/config.json")

# Cell 3 — zip + download:
!cd quasegpt-export && zip -r ../quasegpt-export.zip checkpoint.pt config.json tokenizer.json
# then download quasegpt-export.zip via the Colab file browser,
# unzip locally, and copy the three files into ./model/.
# Verify with: python scripts/verify_model.py
'''

if __name__ == "__main__":
    print(CELLS)
