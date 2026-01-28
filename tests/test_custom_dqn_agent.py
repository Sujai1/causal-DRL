"""Tests for custom DQN agent."""

import numpy as np
import pytest

from causal_fmdp_drl.agents.custom_dqn.agent import DQNAgent, DQNConfig
from causal_fmdp_drl.graphs.causal_graph import CausalGraph


def _make_agent(lambda_reg=0.0, learning_starts=10, train_freq=1, k_global=2):
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 2] = adj[2, 3] = 1
    # Manually set adjacency so k_global = k_global param
    adj_custom = np.zeros((4, 4))
    for i in range(k_global):
        adj_custom[i, k_global] = 1 if k_global < 4 else 0
    graph = CausalGraph(state_vars=["a", "b", "c", "d"], adjacency=adj)
    config = DQNConfig(
        lambda_reg=lambda_reg,
        learning_starts=learning_starts,
        train_freq=train_freq,
        buffer_size=200,
        batch_size=16,
        hidden_dim=32,
    )
    return DQNAgent(obs_dim=4, num_actions=3, config=config, causal_graph=graph)


class TestDQNAgent:
    def test_select_action_returns_valid(self):
        agent = _make_agent()
        obs = np.zeros(4, dtype=np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < 3

    def test_epsilon_decay(self):
        agent = _make_agent()
        eps_start = agent.get_epsilon()
        assert eps_start == 1.0
        agent.step_count = agent.config.eps_decay_steps
        eps_end = agent.get_epsilon()
        assert eps_end == pytest.approx(0.05)

    def test_no_update_before_learning_starts(self):
        agent = _make_agent(learning_starts=100)
        obs = np.zeros(4, dtype=np.float32)
        losses = agent.train_step(obs, 0, 1.0, obs, False)
        assert losses == {}

    def test_update_after_learning_starts(self):
        agent = _make_agent(learning_starts=5, train_freq=1)
        obs = np.zeros(4, dtype=np.float32)
        # Fill buffer past learning_starts
        for _ in range(20):
            losses = agent.train_step(obs, 0, 1.0, obs, False)
        # After enough steps, should get loss dict
        assert "td_loss" in losses

    def test_train_freq_skips_updates(self):
        agent = _make_agent(learning_starts=5, train_freq=4)
        obs = np.zeros(4, dtype=np.float32)
        results = []
        for _ in range(20):
            r = agent.train_step(obs, 0, 1.0, obs, False)
            results.append(r)
        # Only some steps should produce losses
        non_empty = [r for r in results if r]
        empty = [r for r in results if not r]
        assert len(non_empty) > 0
        assert len(empty) > 0

    def test_reg_loss_zero_when_no_reg(self):
        agent = _make_agent(lambda_reg=0.0, learning_starts=5, train_freq=1)
        obs = np.random.randn(4).astype(np.float32)
        for _ in range(20):
            losses = agent.train_step(obs, 0, 1.0, obs, False)
        assert losses.get("reg_loss", 0.0) == 0.0

    def test_reg_loss_nonzero_when_reg_enabled(self):
        agent = _make_agent(lambda_reg=0.01, learning_starts=5, train_freq=1)
        # Use varied observations so features aren't degenerate
        np.random.seed(0)
        for _ in range(50):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            losses = agent.train_step(obs, 0, 1.0, next_obs, False)
        assert losses["reg_loss"] > 0.0
