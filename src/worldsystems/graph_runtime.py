"""Static, inference-only RSSM capture; callers serialize use of each instance."""

import copy
import time

import torch
from torch.nn import functional as F


def unchecked_normal(parameters):
    mean, raw_scale = parameters.chunk(2, -1)
    return torch.distributions.Normal(mean, F.softplus(raw_scale) + 0.1, validate_args=False)


def capture_ready_model(model):
    """Preserve forward arithmetic, move value validation to the request boundary."""
    if model.training:
        raise ValueError("only deterministic eval inference is supported")
    result = copy.deepcopy(model)
    result.distribution = unchecked_normal
    if not all(torch.isfinite(p).all().item() for p in result.parameters()):
        raise ValueError("nonfinite model parameters")
    return result


class StaticGraph:
    def __init__(self, function, observations, actions):
        self.x = observations.clone()
        self.u = actions.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(3):
                function(self.x, self.u)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.output = function(self.x, self.u)
        torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - started) * 1000

    @torch.inference_mode()
    def __call__(self, observations, actions):
        if observations.shape != self.x.shape or actions.shape != self.u.shape:
            raise ValueError("graph shape bucket mismatch")
        if observations.dtype != self.x.dtype or actions.dtype != self.u.dtype:
            raise ValueError("graph precision mismatch")
        # Validation belongs outside capture and is included in public call latency.
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise ValueError("nonfinite graph input")
        self.x.copy_(observations)
        self.u.copy_(actions)
        self.graph.replay()
        result = self.output.clone()
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite graph output")
        return result
