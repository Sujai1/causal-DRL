"""Tests for Q-network with feature hook."""

import torch
import pytest

from causal_fmdp_drl.agents.custom_dqn.network import QNetwork


class TestQNetwork:
    def test_output_shape(self):
        net = QNetwork(obs_dim=10, num_actions=5)
        q = net(torch.randn(32, 10))
        assert q.shape == (32, 5)

    def test_features_shape(self):
        net = QNetwork(obs_dim=10, num_actions=5, hidden_dim=64)
        net(torch.randn(16, 10))
        features = net.get_features()
        assert features.shape == (16, 64)

    def test_forward_with_features(self):
        net = QNetwork(obs_dim=8, num_actions=3, hidden_dim=32)
        q, features = net.forward_with_features(torch.randn(4, 8))
        assert q.shape == (4, 3)
        assert features.shape == (4, 32)

    def test_get_features_before_forward_raises(self):
        net = QNetwork(obs_dim=4, num_actions=2)
        with pytest.raises(RuntimeError, match="Must call forward"):
            net.get_features()

    def test_features_have_gradients(self):
        net = QNetwork(obs_dim=4, num_actions=2)
        obs = torch.randn(8, 4)
        q, features = net.forward_with_features(obs)
        loss = features.sum()
        loss.backward()
        assert net.fc1.weight.grad is not None
