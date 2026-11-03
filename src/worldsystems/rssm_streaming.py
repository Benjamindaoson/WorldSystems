"""Bounded resident RSSM sessions with transactional deadline-checked updates."""

import threading
import time
import uuid

import torch


class RSSMStreaming:
    def __init__(self, model, max_sessions=32, ttl_seconds=300):
        if model.training or max_sessions < 1 or ttl_seconds <= 0:
            raise ValueError("eval model and positive session limits required")
        self.model = model
        self.device = next(model.parameters()).device
        self.max_sessions = max_sessions
        self.ttl = ttl_seconds
        self.sessions = {}
        self.lock = threading.Lock()

    def _reap(self):
        now = time.monotonic()
        for key, (_, _, touched) in list(self.sessions.items()):
            if now - touched > self.ttl:
                del self.sessions[key]

    def _initial(self, observation):
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if x.shape != (1, 3) or not torch.isfinite(x).all():
            raise ValueError("initial observation must be finite [1,3]")
        h = self.model.encoder(x)
        z = self.model.posterior(torch.cat((h, self.model.encoder(x)), -1)).chunk(2, -1)[0]
        return h, z

    @torch.inference_mode()
    def start(self, observation):
        with self.lock:
            self._reap()
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError("session capacity exceeded")
            h, z = self._initial(observation)
            key = uuid.uuid4().hex
            self.sessions[key] = (h, z, time.monotonic())
            return key

    @torch.inference_mode()
    def reset(self, key, observation):
        with self.lock:
            self._reap()
            if key not in self.sessions:
                raise KeyError(key)
            h, z = self._initial(observation)
            self.sessions[key] = (h, z, time.monotonic())

    def close(self, key):
        with self.lock:
            del self.sessions[key]

    @torch.inference_mode()
    def predict(self, key, actions, deadline_ms=1000):
        started = time.monotonic()
        if deadline_ms <= 0:
            raise ValueError("deadline must be positive")
        if not self.lock.acquire(timeout=deadline_ms / 1000):
            raise TimeoutError("session worker deadline exceeded")
        try:
            self._reap()
            h, z, _ = self.sessions[key]
            u = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            if u.ndim != 3 or u.shape[0] != 1 or u.shape[2] != 3 or not 1 <= u.shape[1] <= 64:
                raise ValueError("actions must be [1,T,3], 1<=T<=64")
            if not torch.isfinite(u).all():
                raise ValueError("nonfinite actions")
            predictions = []
            for t in range(u.shape[1]):
                h = self.model.recurrent(torch.cat((z, u[:, t]), -1), h)
                z = self.model.prior(h).chunk(2, -1)[0]
                predictions.append(self.model.decoder(torch.cat((h, z), -1)))
            output = torch.stack(predictions, 1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if not torch.isfinite(output).all():
                raise ValueError("nonfinite prediction")
            if (time.monotonic() - started) * 1000 > deadline_ms:
                raise TimeoutError("prediction expired; session state not committed")
            self.sessions[key] = (h, z, time.monotonic())
            return output
        finally:
            self.lock.release()
