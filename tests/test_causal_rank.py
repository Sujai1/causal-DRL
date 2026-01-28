"""Tests for causal rank regularization penalty."""

import torch
import pytest

from causal_fmdp_drl.regularizers.causal_rank import causal_rank_penalty


class TestCausalRankPenalty:
    def test_zero_when_k_target_exceeds_rank(self):
        features = torch.randn(32, 16)
        penalty = causal_rank_penalty(features, k_target=100)
        assert penalty.item() == 0.0

    def test_zero_when_k_target_equals_min_dim(self):
        features = torch.randn(8, 16)  # min dim is 8
        penalty = causal_rank_penalty(features, k_target=8)
        assert penalty.item() == 0.0

    def test_positive_when_k_target_small(self):
        # Full-rank features should have nonzero penalty for small k_target
        torch.manual_seed(0)
        features = torch.randn(32, 16)
        penalty = causal_rank_penalty(features, k_target=2)
        assert penalty.item() > 0.0

    def test_penalty_decreases_with_larger_k_target(self):
        torch.manual_seed(0)
        features = torch.randn(32, 16)
        p_small = causal_rank_penalty(features, k_target=2)
        p_large = causal_rank_penalty(features, k_target=10)
        assert p_small.item() > p_large.item()

    def test_gradient_flows(self):
        features = torch.randn(16, 8, requires_grad=True)
        penalty = causal_rank_penalty(features, k_target=2)
        penalty.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_k_target_zero_penalizes_all(self):
        torch.manual_seed(0)
        features = torch.randn(32, 16)
        penalty = causal_rank_penalty(features, k_target=0)
        assert penalty.item() > 0.0

    def test_k_target_ge_min_batch_featdim_returns_zero(self):
        """Edge case: k_target >= min(batch_size, feature_dim)."""
        # batch=4, feat_dim=8 -> min=4, k_target=5 should return 0
        features = torch.randn(4, 8)
        penalty = causal_rank_penalty(features, k_target=5)
        assert penalty.item() == 0.0

        # batch=10, feat_dim=3 -> min=3, k_target=3 should return 0
        features = torch.randn(10, 3)
        penalty = causal_rank_penalty(features, k_target=3)
        assert penalty.item() == 0.0
