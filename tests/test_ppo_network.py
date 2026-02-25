"""Tests for PPO actor and critic networks."""

import pytest
import torch
from torch.distributions import Categorical

from causal_fmdp_drl.agents.custom_ppo.network import ActorNetwork, CriticNetwork


class TestCriticNetwork:
    def test_output_shape(self):
        net = CriticNetwork(obs_dim=10, hidden_dim=32)
        obs = torch.randn(8, 10)
        values = net(obs)
        assert values.shape == (8,), f"Expected (8,), got {values.shape}"

    def test_output_shape_single(self):
        net = CriticNetwork(obs_dim=5, hidden_dim=16)
        obs = torch.randn(1, 5)
        values = net(obs)
        assert values.shape == (1,)

    def test_features_stored_after_forward(self):
        net = CriticNetwork(obs_dim=10, hidden_dim=32)
        obs = torch.randn(4, 10)
        net(obs)
        features = net.get_features()
        assert features.shape == (4, 32)

    def test_features_raises_before_forward(self):
        net = CriticNetwork(obs_dim=10, hidden_dim=32)
        with pytest.raises(RuntimeError, match="Must call forward"):
            net.get_features()

    def test_forward_with_features(self):
        net = CriticNetwork(obs_dim=10, hidden_dim=32)
        obs = torch.randn(4, 10)
        values, features = net.forward_with_features(obs)
        assert values.shape == (4,)
        assert features.shape == (4, 32)

    def test_features_are_post_relu(self):
        """Features should be non-negative (post-ReLU)."""
        net = CriticNetwork(obs_dim=10, hidden_dim=64)
        obs = torch.randn(16, 10)
        _, features = net.forward_with_features(obs)
        assert (features >= 0).all(), "Features should be post-ReLU (non-negative)"

    def test_layernorm_variant(self):
        net = CriticNetwork(obs_dim=10, hidden_dim=32, use_layernorm=True)
        assert hasattr(net, "ln1")
        assert hasattr(net, "ln2")
        obs = torch.randn(4, 10)
        values, features = net.forward_with_features(obs)
        assert values.shape == (4,)
        assert features.shape == (4, 32)

    def test_no_layernorm_by_default(self):
        net = CriticNetwork(obs_dim=10, hidden_dim=32)
        assert not hasattr(net, "ln1") or not net.use_layernorm


class TestActorNetwork:
    def test_output_shape(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=32)
        obs = torch.randn(8, 10)
        logits = net(obs)
        assert logits.shape == (8, 5), f"Expected (8, 5), got {logits.shape}"

    def test_features_stored_after_forward(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=32)
        obs = torch.randn(4, 10)
        net(obs)
        features = net.get_features()
        assert features.shape == (4, 32)

    def test_forward_with_features(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=32)
        obs = torch.randn(4, 10)
        logits, features = net.forward_with_features(obs)
        assert logits.shape == (4, 5)
        assert features.shape == (4, 32)

    def test_get_distribution_returns_categorical(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=32)
        obs = torch.randn(4, 10)
        dist = net.get_distribution(obs)
        assert isinstance(dist, Categorical)
        assert dist.probs.shape == (4, 5)

    def test_distribution_samples_valid_actions(self):
        net = ActorNetwork(obs_dim=10, num_actions=3, hidden_dim=32)
        obs = torch.randn(100, 10)
        dist = net.get_distribution(obs)
        samples = dist.sample()
        assert samples.shape == (100,)
        assert (samples >= 0).all() and (samples < 3).all()

    def test_distribution_log_probs_finite(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=32)
        obs = torch.randn(4, 10)
        dist = net.get_distribution(obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        assert torch.isfinite(log_probs).all()

    def test_layernorm_variant(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=32, use_layernorm=True)
        obs = torch.randn(4, 10)
        logits, features = net.forward_with_features(obs)
        assert logits.shape == (4, 5)
        assert features.shape == (4, 32)

    def test_features_are_post_relu(self):
        net = ActorNetwork(obs_dim=10, num_actions=5, hidden_dim=64)
        obs = torch.randn(16, 10)
        _, features = net.forward_with_features(obs)
        assert (features >= 0).all()
