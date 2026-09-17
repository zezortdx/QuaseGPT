"""Internal local model runtime for Rails. NOT the product interface.

Endpoints:
  GET  /health
  POST /generate
  POST /generate/stream  (Server-Sent Events)

Security: only generation parameters are accepted. Model paths come from
trusted app config (env), never from the request. No shell/Python execution,
no filesystem paths, no checkpoint loading from web input.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, field_validator  # noqa: E402

from quasegpt import generate, generate_stream, validate_params  # noqa: E402
from runtime import load_runtime  # noqa: E402

from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _state["runtime"] = load_runtime()
    except Exception as e:  # keep server up so /health reports the problem
        _state["error"] = str(e)
        print(f"[quasegpt] failed to load model: {e}")
    finally:
        _state["ready"].set()
    yield


app = FastAPI(title="QuaseGPT runtime", version="0.1.0", lifespan=lifespan)

_state: dict = {"runtime": None, "error": None, "ready": threading.Event()}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = None
    seed: int | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("prompt must be non-empty")
        if len(v) > 32_000:
            raise ValueError("prompt too long")
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def _tokens(cls, v: int) -> int:
        if not (1 <= v <= 1024):
            raise ValueError("max_new_tokens must be 1..1024")
        return v

    @field_validator("temperature")
    @classmethod
    def _temp(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be 0.0..2.0")
        return v

    @field_validator("top_k")
    @classmethod
    def _topk(cls, v) -> int | None:
        if v is not None and not (1 <= v <= 1000):
            raise ValueError("top_k must be 1..1000")
        return v

    @field_validator("top_p")
    @classmethod
    def _topp(cls, v) -> float | None:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("top_p must be 0 < top_p <= 1.0")
        return v


def _rt():
    rt = _state.get("runtime")
    if rt is None:
        raise HTTPException(status_code=503, detail=_state.get("error") or "model not loaded")
    return rt


@app.get("/health")
def health():
    rt = _state.get("runtime")
    return {
        "status": "ok" if rt is not None else "error",
        "model_loaded": rt is not None,
        "device": str(rt.device) if rt else None,
        "params": rt.param_count if rt else None,
        "checkpoint": rt.checkpoint_path if rt else None,
        "error": _state.get("error"),
    }


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    rt = _rt()
    try:
        validate_params(req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.top_p)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        with torch.inference_mode():
            text, n = generate(rt.model, rt.tokenizer, req.prompt,
                               max_new_tokens=req.max_new_tokens,
                               temperature=req.temperature, top_k=req.top_k,
                               top_p=req.top_p, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")
    return {"text": text, "generated_tokens": n}


@app.post("/generate/stream")
def generate_stream_endpoint(req: GenerateRequest):
    rt = _rt()
    try:
        validate_params(req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.top_p)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    def event_iter():
        count = [0]
        try:
            with torch.inference_mode():
                for delta in generate_stream(
                        rt.model, rt.tokenizer, req.prompt,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature, top_k=req.top_k,
                        top_p=req.top_p, seed=req.seed):
                    count[0] += 1
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
        except asyncio.CancelledError:
            return  # client disconnected: stop generating
        except GeneratorExit:
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("Ready on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
