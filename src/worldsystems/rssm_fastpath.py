"""Deterministic free-running RSSM mean path for inference-only execution."""

import torch


def predict_prior_means(model, observations: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
    """Compute the same evaluation outputs without constructing unused Normal scales."""
    if model.training:
        raise ValueError("prior-mean fastpath requires eval mode")
    if observations.ndim != 3 or actions.ndim != 3 or observations.shape != (
            actions.shape[0], actions.shape[1] + 1, 3) or actions.shape[-1] != 3:
        raise ValueError("invalid RSSM input shapes")
    current = model.encoder(observations[:, 0])
    encoded_initial = model.encoder(observations[:, 0])
    latent = model.posterior(torch.cat((current, encoded_initial), -1)).chunk(2, -1)[0]
    predictions = []
    for t in range(actions.shape[1]):
        current = model.recurrent(torch.cat((latent, actions[:, t]), -1), current)
        latent = model.prior(current).chunk(2, -1)[0]
        predictions.append(model.decoder(torch.cat((current, latent), -1)))
    return torch.stack(predictions, 1)
