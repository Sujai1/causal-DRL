"""Tests for custom PPO agent."""

import numpy as np
import pytest
import torch

from causal_fmdp_drl.agents.custom_ppo.agent import PPOAgent, PPOConfig


def _make_agent(**kwargs) -> PPOAgent:
    """Create a PPOAgent with small defaults for testing."""
    defaults = dict(
        n_steps=64,
        batch_size=16,
        n_epochs=2,
        hidden_dim=32,
        lr=3e-4,
    )
    defaults.update(kwargs)
    config = PPOConfig(**defaults)
    return PPOAgent(obs_dim=4, num_actions=3, config=config)


def _fill_buffer(agent: PPOAgent, n_steps: int = None):
    """Fill the agent's buffer with random transitions."""
    if n_steps is None:
        n_steps = agent.config.n_steps
    agent.buffer.reset()
    for _ in range(n_steps):
        obs = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        reward = np.random.randn()
        terminated = np.random.random() < 0.05
        truncated = (not terminated) and (np.random.random() < 0.05)
        agent.buffer.add(obs, action, reward, terminated, truncated, log_prob, value)

    # Compute GAE
    obs = np.random.randn(4).astype(np.float32)
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        last_value = agent.critic(obs_t).item()

    buf = agent.buffer
    n = buf.size
    last_terminated = bool(buf.terminateds[n - 1]) if n > 0 else False
    last_truncated = bool(buf.truncateds[n - 1]) if n > 0 else False

    buf.compute_returns_and_advantages(
        last_value=last_value,
        last_terminated=last_terminated,
        last_truncated=last_truncated,
        gamma=agent.config.gamma,
        gae_lambda=agent.config.gae_lambda,
    )


class TestSelectAction:
    def test_returns_valid_action(self):
        agent = _make_agent()
        obs = np.zeros(4, dtype=np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert 0 <= action < 3
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_eval_mode_deterministic(self):
        """Eval mode should always return the same (greedy) action for same input."""
        agent = _make_agent()
        torch.manual_seed(42)
        obs = np.ones(4, dtype=np.float32)
        actions = set()
        for _ in range(20):
            action, _, _ = agent.select_action(obs, eval_mode=True)
            actions.add(action)
        assert len(actions) == 1, "Eval mode should be deterministic"

    def test_explore_mode_varies(self):
        """Exploration mode should sometimes produce different actions."""
        agent = _make_agent()
        obs = np.zeros(4, dtype=np.float32)
        actions = set()
        for _ in range(100):
            action, _, _ = agent.select_action(obs)
            actions.add(action)
        # With 3 actions and random init, we expect diversity
        assert len(actions) >= 2, "Exploration should produce varied actions"


class TestUpdate:
    def test_update_returns_metric_keys(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent()
        _fill_buffer(agent)
        metrics = agent.update()

        expected_keys = [
            "policy_loss", "value_loss", "entropy", "clip_fraction",
            "approx_kl", "explained_variance", "adv_mean", "adv_std",
            "value_pred_mean", "return_mean", "grad_norm", "reg_loss",
            "epochs_run",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"

    def test_update_finite_values(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent()
        _fill_buffer(agent)
        metrics = agent.update()

        for key, val in metrics.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"

    def test_reg_loss_zero_when_disabled(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(lambda_reg=0.0)
        _fill_buffer(agent)
        metrics = agent.update()
        assert metrics["reg_loss"] == 0.0

    def test_empty_buffer_returns_empty(self):
        agent = _make_agent()
        # Don't fill buffer
        metrics = agent.update()
        assert metrics == {}


class TestKLEarlyStopping:
    def test_kl_early_stopping(self):
        """With a very low target_kl, training should stop early."""
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(n_epochs=100, target_kl=1e-10)
        _fill_buffer(agent)
        metrics = agent.update()

        # Should stop well before 100 epochs
        assert metrics["epochs_run"] < 100, (
            f"Expected early stopping but ran {metrics['epochs_run']} epochs"
        )

    def test_no_kl_stopping_without_target(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(n_epochs=3, target_kl=None)
        _fill_buffer(agent)
        metrics = agent.update()
        assert metrics["epochs_run"] == 3


class TestValueClipping:
    def test_value_clipping_runs(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(clip_range_vf=0.2)
        _fill_buffer(agent)
        metrics = agent.update()
        assert "value_loss" in metrics
        assert np.isfinite(metrics["value_loss"])


class TestAdvNormalization:
    def test_minibatch_normalization(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(adv_norm="minibatch")
        _fill_buffer(agent)
        metrics = agent.update()
        assert np.isfinite(metrics["policy_loss"])

    def test_rollout_normalization(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(adv_norm="rollout")
        _fill_buffer(agent)
        metrics = agent.update()
        assert np.isfinite(metrics["policy_loss"])

    def test_no_normalization(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(adv_norm="none")
        _fill_buffer(agent)
        metrics = agent.update()
        assert np.isfinite(metrics["policy_loss"])


class TestSeparateOptimizers:
    def test_separate_optimizers(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(separate_optimizers=True)
        assert agent.optimizer is None
        assert agent.actor_optimizer is not None
        assert agent.critic_optimizer is not None
        _fill_buffer(agent)
        metrics = agent.update()
        assert np.isfinite(metrics["policy_loss"])


class TestLayerNorm:
    def test_layernorm_agent(self):
        np.random.seed(0)
        torch.manual_seed(0)
        agent = _make_agent(use_layernorm=True)
        assert agent.actor.use_layernorm
        assert agent.critic.use_layernorm
        _fill_buffer(agent)
        metrics = agent.update()
        assert np.isfinite(metrics["policy_loss"])
