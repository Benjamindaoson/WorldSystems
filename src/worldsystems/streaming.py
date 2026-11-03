"""Isolated recurrent sessions for the compact reference model."""

import time
import uuid

import torch

from .model import TinyWorld


class StreamingRuntime:
    def __init__(self, model: TinyWorld, max_sessions: int = 32, ttl_seconds: float = 300):
        self.model = model.eval()
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, tuple[torch.Tensor | None, float, int]] = {}

    def start(self) -> str:
        self.reap()
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("session capacity exceeded")
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = (None, time.monotonic(), 0)
        return session_id

    def reset(self, session_id: str) -> None:
        if session_id not in self.sessions:
            raise KeyError(session_id)
        _, _, version = self.sessions[session_id]
        self.sessions[session_id] = (None, time.monotonic(), version + 1)

    def close(self, session_id: str) -> None:
        del self.sessions[session_id]

    def reap(self) -> None:
        now = time.monotonic()
        for key, (_, touched, _) in list(self.sessions.items()):
            if now - touched > self.ttl_seconds:
                del self.sessions[key]

    @torch.inference_mode()
    def prepare_step(self, session_id: str, frame: torch.Tensor,
                     action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        self.reap()
        if session_id not in self.sessions:
            raise KeyError(session_id)
        state, _, version = self.sessions[session_id]
        if frame.shape != (1, 1, 3, self.model.image_size, self.model.image_size):
            raise ValueError("stream frame must be [1,1,3,H,W]")
        if action.shape != (1, 1, self.model.action_dim):
            raise ValueError("stream action must be [1,1,A]")
        output, new_state = self.model(frame, action, state)
        return output, new_state.detach(), version

    def commit_step(self, session_id: str, state: torch.Tensor, version: int) -> None:
        self.reap()
        if session_id not in self.sessions or self.sessions[session_id][2] != version:
            raise RuntimeError("session changed before prediction commit")
        self.sessions[session_id] = (state, time.monotonic(), version + 1)

    def step(self, session_id: str, frame: torch.Tensor,
             action: torch.Tensor) -> torch.Tensor:
        output, state, version = self.prepare_step(session_id, frame, action)
        self.commit_step(session_id, state, version)
        return output
