import torch
from torch import nn

from .data import validate_batch


class TinyWorld(nn.Module):
    """Predict next RGB frame from each frame and action, using causal GRU context."""

    def __init__(self, image_size: int = 8, action_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.image_size = image_size
        self.action_dim = action_dim
        pixels = 3 * image_size * image_size
        self.frame_encoder = nn.Sequential(nn.Linear(pixels, hidden_dim), nn.SiLU())
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.temporal = nn.GRU(2 * hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, pixels)

    def forward(self, video: torch.Tensor, actions: torch.Tensor,
                state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        validate_batch({"video": video, "actions": actions, "target": video}, self.action_dim)
        _, _, _, h, w = video.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError("image resolution mismatch")
        return self.forward_compute(video, actions, state)

    def forward_compute(self, video: torch.Tensor, actions: torch.Tensor,
                        state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Validated inputs only: static-shape compute region for CUDA Graph experiments."""
        b, t, _, h, w = video.shape
        encoded = self.frame_encoder(video.reshape(b, t, -1))
        action = self.action_encoder(actions)
        features, new_state = self.temporal(torch.cat((encoded, action), dim=-1), state)
        predicted = self.decoder(features).reshape(b, t, 3, h, w).sigmoid()
        return predicted, new_state
