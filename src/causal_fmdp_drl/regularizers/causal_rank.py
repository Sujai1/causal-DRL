"""Causal rank regularization."""

import torch


def causal_rank_penalty(
    features: torch.Tensor,
    k_target: int,
) -> torch.Tensor:
    """Penalize singular values of the feature matrix beyond rank k_target.

    R_causal = sum_{j > k_target} sigma_j^2

    where sigma_j are the singular values of the mean-centered feature matrix.

    Args:
        features: (batch_size, feature_dim) feature matrix.
        k_target: Target rank (K_causal). Singular values at indices
            >= k_target are penalized.

    Returns:
        Scalar penalty tensor (0 if k_target >= number of singular values).
    """
    max_rank = min(features.shape[0], features.shape[1])
    if k_target >= max_rank:
        return torch.tensor(0.0, device=features.device)

    features_centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(features_centered)

    return (singular_values[k_target:] ** 2).sum()
