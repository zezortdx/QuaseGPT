# QuaseGPT

A tiny GPT trained from scratch.

Almost intelligent.

Our own decoder-only Transformer, trained from scratch in PyTorch, served
locally with a Ruby on Rails chat UI. **No external LLM APIs.**
Google Colab is optional and only used for GPU-heavy training.

```
Browser → Ruby on Rails (chat UI, persistence)
            → local QuaseGPT inference runtime (FastAPI, ml/server.py)
                → our trained PyTorch checkpoint (model/checkpoint.pt)
```

## Quickstart (local, no Colab)

First-time setup:

```sh
git clone https://github.com/zezortdx/QuaseGPT.git
cd QuaseGPT
python3 -m venv .venv
.venv/bin/pip install -r ml/requirements.txt
bundle install
bin/rails db:prepare
```

Then install the trained weights at `model/checkpoint.pt`
(`config.json` + `tokenizer.json` are already in the repo) and verify:

```sh
.venv/bin/python scripts/verify_model.py   # success = VERIFY OK
```

Daily usage:

```sh
./quasegpt
```

This checks your setup, starts Rails + the local runtime, and opens
the chat UI at http://localhost:3000 (health:
http://127.0.0.1:8000/health). Ctrl+C stops everything.

On macOS you can also double-click `QuaseGPT.command`.

Developers who want the lower-level pieces can run `bin/dev` directly
(same two processes, no preflight checks or auto-open); set
`QUASEGPT_NO_BROWSER=1` to skip the browser auto-open.

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

QuaseGPT 39M — trained from scratch as a **base language model** on
TinyStories-style text (**not instruction-tuned**): vocab 2000 ·
context 256 · 12 layers / 8 heads / d512, tied weights, ≈38.98M params,
trained to step 19999.

`model/config.json` plus the matching `model/tokenizer.json` are committed
and describe exactly this trained model, so the app boots and tests run.
The trained weights themselves (`model/checkpoint.pt`) are intentionally
NOT in Git — without them the runtime loads with random weights and says
so honestly (`Checkpoint: MISSING (random weights!)`, health `error`).

To install the real weights locally (the checkpoint lives outside the git
tree — e.g. a future GitHub Release asset — never commit it):

1. Follow `training/colab/README.md` → download `quasegpt-export.zip`.
2. `unzip quasegpt-export.zip -d quasegpt-export`
3. `cp quasegpt-export/{checkpoint.pt,config.json,tokenizer.json} model/`
4. `python scripts/verify_model.py` (loads + 16-token smoke generation)
5. Restart `bin/dev`.

QuaseGPT is a **base language model, not instruction-tuned**: it continues
text plausibly rather than following instructions like ChatGPT. Chat behavior
depends on training; the template lives in one replaceable place
(`ml/quasegpt/prompt.py` + `lib/quase_gpt_prompt.rb`).

## Running

`bin/dev` starts both processes: Rails on `:3000`, inference runtime on
`:8000`. Without a checkpoint the app still boots — the runtime reports
`MISSING (random weights!)` and the UI shows a failure note instead of
fake text.

```sh
python ml/chat.py                 # CLI inference: "Once upon a time", empty line quits
python scripts/smoke_test.py      # live server health + generate check
```

Configuration (all optional):

| Variable | Default | Effect |
|---|---|---|
| `QUASEGPT_INFERENCE_URL` | `http://127.0.0.1:8000` | where Rails finds the runtime (`bin/dev` derives the runtime port from it) |
| `QUASEGPT_MODEL_DIR` | `./model` | where the runtime loads `config.json` / `tokenizer.json` / `checkpoint.pt` |
| `QUASEGPT_DEVICE` | `auto` | `auto` picks CUDA → MPS → CPU; force with `cuda`, `mps`, or `cpu` |
| `QUASEGPT_BLOCK_SIZE` | `256` | context budget mirror on the Rails side |

If port 3000 is taken by another app, run Rails on another port:
`bin/rails server -p 3001` (start the runtime separately with
`PORT=8000 .venv/bin/python ml/server.py` instead of `bin/dev`).

## Training

Train your own weights (GPU recommended; CPU/MPS work but are slow):

```sh
python ml/prepare_data.py --input data/corpus.txt --out-dir data \
    --tokenizer model/tokenizer.json --train-tokenizer --vocab-size 2000
python ml/train.py --config configs/training.json
```

Resume after an interruption — continues from checkpoint step + 1 toward
`max_steps` (final target, not an additional count):

```sh
python ml/train.py --config configs/training.json --resume model/checkpoint.pt
```

Full GPU workflow (dataset, smoke test, Drive persistence, export):
`training/colab/README.md`.

## Limitations

- Base model: great at continuing TinyStories-style text, bad at
  following instructions. That's the training, not a bug.
- Small: 39M params, 256-token context. Don't expect essays or facts.
- No safety tuning. Don't deploy the raw model to strangers.

## License

MIT — see [LICENSE](LICENSE).

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
