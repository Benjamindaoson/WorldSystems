"""Bounded loopback serving for a verified real RSSM checkpoint."""

import asyncio
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from .rssm_adapter import RSSMAdapter


class RSSMRequest(BaseModel):
    observations: list
    actions: list
    deadline_ms: int = Field(default=1000, le=30000)


def create_rssm_app(repository: str, weights: str, device: str = "cpu") -> FastAPI:
    adapter = RSSMAdapter(repository, weights, device)
    app = FastAPI(title="WorldSystems RSSM inference")
    app.state.adapter = adapter
    app.state.busy = False

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.method == "POST":
            try:
                claimed = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "invalid content length"}, status_code=400)
            if claimed > 128 * 1024:
                return JSONResponse({"detail": "request too large"}, status_code=413)
            chunks = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 128 * 1024:
                    return JSONResponse({"detail": "request too large"}, status_code=413)
                chunks.append(chunk)
            body = b"".join(chunks)
            request._body = body
            async def replay_body():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = replay_body
        return await call_next(request)

    @app.get("/health")
    def health():
        loaded = app.state.adapter is not None
        return {"status": "ready" if loaded else "unloaded", "model_family": "WorldDynamics RSSM"}

    @app.post("/unload")
    async def unload():
        if app.state.busy:
            raise HTTPException(429, "inference busy")
        app.state.adapter = None
        if device == "cuda":
            torch.cuda.empty_cache()
        return {"status": "unloaded"}

    @app.post("/reload")
    async def reload():
        if app.state.busy:
            raise HTTPException(429, "inference busy")
        app.state.adapter = RSSMAdapter(repository, weights, device)
        return {"status": "ready", "checkpoint_sha256": app.state.adapter.weight_sha256}

    @app.post("/predict")
    async def predict(request: RSSMRequest):
        if request.deadline_ms <= 0:
            raise HTTPException(400, "invalid deadline")
        if len(request.actions) > 8 or len(request.observations) > 8:
            raise HTTPException(413, "batch too large")
        if app.state.adapter is None:
            raise HTTPException(503, "model unloaded")
        if app.state.busy:
            raise HTTPException(429, "inference busy")
        started = time.perf_counter()
        try:
            observations = torch.tensor(request.observations, dtype=torch.float32)
            actions = torch.tensor(request.actions, dtype=torch.float32)
            if actions.ndim != 3 or not 1 <= actions.shape[1] <= 64 or not len(actions) or observations.shape != (
                    actions.shape[0], actions.shape[1] + 1, 3) or actions.shape[-1] != 3:
                raise ValueError("expected observations [B,T+1,3], actions [B,T,3], T<=64")
            if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
                raise ValueError("inputs must be finite")
        except (ValueError, TypeError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        app.state.busy = True
        adapter_snapshot = app.state.adapter

        @torch.inference_mode()
        def work():
            output = adapter_snapshot.predict(observations, actions)
            if device == "cuda":
                torch.cuda.synchronize()
            return output.cpu().tolist()

        task = asyncio.create_task(asyncio.to_thread(work))

        def release(done):
            app.state.busy = False
            if not done.cancelled():
                done.exception()

        task.add_done_callback(release)
        try:
            prediction = await asyncio.wait_for(asyncio.shield(task), request.deadline_ms / 1000)
        except TimeoutError:
            raise HTTPException(504, "inference deadline exceeded") from None
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(503, f"GPU memory exhausted: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(503, f"model failed: {exc}") from exc
        return {"prediction": prediction, "valid": True,
                "server_processing_ms": (time.perf_counter() - started) * 1000,
                "checkpoint_sha256": adapter_snapshot.weight_sha256}

    return app
