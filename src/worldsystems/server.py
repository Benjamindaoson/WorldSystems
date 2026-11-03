import asyncio
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from .config import Config
from .runtime import load_model
from .streaming import StreamingRuntime


class PredictionRequest(BaseModel):
    video: list
    actions: list
    session_id: str | None = None
    deadline_ms: int = Field(default=1000, le=30000)


MAX_BODY_BYTES = 256 * 1024
MAX_VIDEO_FRAMES = 64


def _validate_size(request: PredictionRequest) -> None:
    if len(request.video) > MAX_VIDEO_FRAMES or len(request.actions) > MAX_VIDEO_FRAMES:
        raise HTTPException(413, "too many frames")


def create_app(cfg: Config, checkpoint: str) -> FastAPI:
    model = load_model(cfg, checkpoint)
    app = FastAPI(title="WorldSystems local inference")
    stream = StreamingRuntime(model)
    app.state.model = model
    app.state.stream = stream
    busy = {"active": False}

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                claimed = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "invalid content length"}, status_code=400)
            if claimed > MAX_BODY_BYTES:
                return JSONResponse({"detail": "request too large"}, status_code=413)
            body = await request.body()
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse({"detail": "request too large"}, status_code=413)
            async def replay_body():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = replay_body
        return await call_next(request)

    async def run_bounded(function, *args, deadline_ms: int):
        if deadline_ms <= 0:
            raise HTTPException(400, "invalid deadline")
        if busy["active"]:
            raise HTTPException(429, "inference busy")
        busy["active"] = True
        @torch.inference_mode()
        def work():
            return function(*args)

        task = asyncio.create_task(asyncio.to_thread(work))

        def finished(done):
            busy["active"] = False
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=deadline_ms / 1000)
        except TimeoutError:
            raise HTTPException(504, "inference deadline exceeded") from None

    @app.get("/health")
    def health():
        return {"status": "ready", "model_family": "tiny-action-temporal"}

    @app.post("/sessions")
    def session():
        try:
            return {"session_id": stream.start()}
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.delete("/sessions/{session_id}")
    def close_session(session_id: str):
        try:
            stream.close(session_id)
        except KeyError:
            raise HTTPException(404, "unknown session") from None
        return {"closed": True}

    @app.post("/sessions/{session_id}/step")
    async def stream_step(session_id: str, request: PredictionRequest):
        _validate_size(request)
        try:
            started = time.perf_counter()
            frame = torch.tensor(request.video, dtype=torch.float32, device=cfg.device)
            action = torch.tensor(request.actions, dtype=torch.float32, device=cfg.device)
            compute_start = time.perf_counter()
            output, state, version = await run_bounded(
                stream.prepare_step, session_id, frame, action,
                deadline_ms=request.deadline_ms)
            compute_ms = (time.perf_counter() - compute_start) * 1000
            stream.commit_step(session_id, state, version)
            result = {"prediction": output.cpu().tolist(), "valid": True,
                      "compute_and_dispatch_ms": compute_ms}
            result["server_processing_ms"] = (time.perf_counter() - started) * 1000
            return result
        except KeyError:
            raise HTTPException(404, "unknown or expired session") from None
        except (ValueError, RuntimeError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/predict")
    async def predict(request: PredictionRequest):
        _validate_size(request)
        if request.session_id and request.session_id not in stream.sessions:
            raise HTTPException(404, "unknown session")
        try:
            started = time.perf_counter()
            video = torch.tensor(request.video, dtype=torch.float32, device=cfg.device)
            actions = torch.tensor(request.actions, dtype=torch.float32, device=cfg.device)
            compute_start = time.perf_counter()
            output, _ = await run_bounded(model, video, actions,
                                          deadline_ms=request.deadline_ms)
            compute_ms = (time.perf_counter() - compute_start) * 1000
            result = {"prediction": output.cpu().tolist(), "valid": True,
                      "compute_and_dispatch_ms": compute_ms}
            result["server_processing_ms"] = (time.perf_counter() - started) * 1000
            return result
        except (ValueError, TypeError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
