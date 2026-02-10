"""Causal rank regularization.

All penalties operate on centered feature matrices (mean subtracted per column).
Centering changes effective rank by at most 1; this is intentional and consistent
across all methods.
"""

import torch


def nuclear_norm_penalty(features: torch.Tensor) -> torch.Tensor:
    """Nuclear norm of centered and normalized feature matrix.

    Args:
        features: (batch_size, feature_dim) feature matrix.

    Returns:
        Scalar penalty tensor (sum of all singular values).
    """
    features_centered = features - features.mean(dim=0, keepdim=True)
    features_normalized = features_centered / (features.shape[0] ** 0.5)
    singular_values = torch.linalg.svdvals(features_normalized)
    return singular_values.sum()


def causal_rank_penalty(
    features: torch.Tensor,
    k_target: int,
) -> torch.Tensor:
    """Penalize singular values of the feature matrix beyond rank k_target.

    R_causal = sum_{j > k_target} sigma_j^2

    where sigma_j are the singular values of the centered and normalized
    feature matrix.

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
    features_normalized = features_centered / (features.shape[0] ** 0.5)
    singular_values = torch.linalg.svdvals(features_normalized)

    return (singular_values[k_target:] ** 2).sum()


def relative_tail_energy(
    features: torch.Tensor,
    k_target: int,
    use_topk: bool = False,
) -> torch.Tensor:
    """Relative tail energy: fraction of variance in tail singular values.

    R_relative = sum_{j >= k_target} sigma_j^2 / (sum_j sigma_j^2 + eps)

    This is scale-invariant - the network cannot game the penalty by scaling
    features up or down. Values range from 0 (all energy in top-k) to 1
    (all energy in tail).

    Features are centered (mean subtracted per column) before SVD. This changes
    effective rank by at most 1.

    Args:
        features: (batch_size, feature_dim) feature matrix. Can be larger than
            a single batch if using a feature buffer for stable SVD estimates.
        k_target: Target rank. Singular values at indices >= k_target are
            considered "tail" energy.
        use_topk: If True, use top-K eigenvalue computation via lobpcg instead
            of full SVD. Only beneficial for large matrices (d > 256) with small K.
            Default False (full SVD is faster for typical d=64-128).

    Returns:
        Scalar tensor in [0, 1]. Returns 0 if k_target >= min(n, d).
    """
    n, d = features.shape
    max_rank = min(n, d)

    # If k_target is at or beyond the maximum possible rank, no tail to penalize
    if k_target >= max_rank:
        return torch.tensor(0.0, device=features.device, requires_grad=False)

    # Center and normalize for batch-size invariance
    features_centered = features - features.mean(dim=0, keepdim=True)
    features_normalized = features_centered / (n ** 0.5)

    # Total energy (Frobenius norm squared) - always cheap O(nd)
    total_energy = (features_normalized ** 2).sum()

    if use_topk and k_target < max_rank // 2:
        # Top-K eigenvalue path via lobpcg
        # Only beneficial for large matrices (d > 256) with small K.
        # For d=64, full SVD is ~10x faster due to lobpcg overhead.
        #
        # tail_energy = ||Φ̃||_F² - Σ_{j≤K} σ_j²
        # Eigenvalues of Φ̃ᵀΦ̃ (or Φ̃Φ̃ᵀ) are σ²

        # Choose the smaller Gram matrix
        if n <= d:
            gram = features_normalized @ features_normalized.T  # n×n
        else:
            gram = features_normalized.T @ features_normalized  # d×d

        matrix_size = gram.shape[0]
        k_to_compute = min(k_target, matrix_size - 1)

        if k_to_compute <= 0:
            top_k_energy = torch.tensor(0.0, device=features.device)
        else:
            X = torch.randn(matrix_size, k_to_compute, device=features.device, dtype=features.dtype)
            try:
                eigenvalues, _ = torch.lobpcg(gram, k=k_to_compute, X=X, largest=True, niter=20)
                top_k_energy = eigenvalues.sum()
            except RuntimeError:
                # Fallback to full eigendecomposition if lobpcg fails
                eigenvalues = torch.linalg.eigvalsh(gram)
                # eigvalsh returns ascending order, take last k_to_compute (largest)
                top_k_energy = eigenvalues[-k_to_compute:].sum()

        tail_energy = total_energy - top_k_energy
        tail_energy = torch.clamp(tail_energy, min=0.0)

    else:
        # Full SVD path (default, faster for d < 256)
        singular_values = torch.linalg.svdvals(features_normalized)
        tail_energy = (singular_values[k_target:] ** 2).sum()

    return tail_energy / (total_energy + 1e-8)
