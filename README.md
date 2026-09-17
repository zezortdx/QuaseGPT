# QuaseGPT — Almost intelligent.

Our own decoder-only Transformer, trained from scratch in PyTorch, served
locally with a Ruby on Rails chat UI. **No external LLM APIs.**
Google Colab is optional and only used for GPU-heavy training.

```
Browser → Ruby on Rails (chat UI, persistence)
            → local QuaseGPT inference runtime (FastAPI, ml/server.py)
                → our trained PyTorch checkpoint (model/checkpoint.pt)
```

## Quickstart (local, no Colab)

```sh
bundle install
pip install -r ml/requirements.txt        # or: python3 -m venv .venv && .venv/bin/pip install -r ml/requirements.txt
bin/rails db:prepare
bin/dev
```

Then open:

- Chat UI: http://localhost:3000
- Runtime health: http://127.0.0.1:8000/health (`QUASEGPT_INFERENCE_URL` overrides the Rails side)

Try the definitive test with Colab closed: send **“Once upon a time”** —
the reply must come from the local checkpoint.

## How it fits together

| Path | Responsibility |
|---|---|
| `ml/quasegpt/` | The actual model: `model.py`, `blocks.py`, `attention.py`, `config.py`, `tokenizer.py`, `checkpoint.py`, `generation.py`, `device.py`, `prompt.py`. Shared by training, CLI, and server — never forked. |
| `ml/server.py` | Local inference HTTP service: `GET /health`, `POST /generate`, `POST /generate/stream` (SSE). Only generation params accepted; model paths come from env, never requests. |
| `ml/chat.py` | CLI smoke test: `python ml/chat.py` loads config + tokenizer + checkpoint and generates locally. |
| `ml/train.py`, `ml/prepare_data.py` | Training loop and dataset prep. Same code runs locally or in Colab. |
| `model/` | Runtime artifacts: `config.json` + `tokenizer.json` (committed), `checkpoint.pt` (exported, gitignored). |
| `app/`, `config/`, `db/` | Rails chat UI: sidebar, conversation history, composer, persistence, SSE streaming. No PyTorch here. |
| `lib/quase_gpt_client.rb` | Rails → runtime HTTP client with graceful failure handling. |
| `lib/quase_gpt_prompt.rb` | `User:/Assistant:` chat template + context budgeting (mirrors `ml/quasegpt/prompt.py`; the runtime enforces the exact token limit). |
| `training/colab/README.md` | Colab GPU training + export workflow. |

## Model status

`model/config.json` (vocab 2000 · context 256 · 6 layers / 6 heads / d384,
tied weights, ≈11.5M params) plus the matching `model/tokenizer.json` are
committed so the app boots and tests run. **There is no `checkpoint.pt`
yet** — until you export one, the runtime loads with random weights and says
so honestly (`Checkpoint: MISSING (random weights!)`, health `error`).

To install the real weights (the only remaining artifact step):

1. Follow `training/colab/README.md` → download `quasegpt-export.zip`.
2. `unzip quasegpt-export.zip -d quasegpt-export`
3. `cp quasegpt-export/{checkpoint.pt,config.json,tokenizer.json} model/`
4. `python scripts/verify_model.py` (loads + 16-token smoke generation)
5. Restart `bin/dev`.

QuaseGPT is a **base language model, not instruction-tuned**: it continues
text plausibly rather than following instructions like ChatGPT. Chat behavior
depends on training; the template lives in one replaceable place
(`ml/quasegpt/prompt.py` + `lib/quase_gpt_prompt.rb`).

## Tests

```sh
.venv/bin/python -m pytest ml/tests -q   # Python runtime (no checkpoint/GPU needed)
bin/rails test                           # Rails (runtime stubbed, no server needed)
```

CI (`.github/workflows/ci.yml`) runs both, plus Brakeman/bundler-audit,
importmap audit, and RuboCop — all without checkpoints, GPUs, or datasets.

## Scripts

- `python scripts/verify_model.py` — config + tokenizer + checkpoint load check.
- `python scripts/smoke_test.py` — live server health + generate check.
- `python scripts/export_from_colab.py` — prints the Colab export cells.

## Security notes

No arbitrary code/shell execution, no user-selected checkpoint paths, no
checkpoint upload endpoint, generation params validated and capped
(`max_new_tokens` ≤ 1024 runtime-side, 128 from the UI; prompts ≤ 32000
chars). Model files come from trusted local config only.
