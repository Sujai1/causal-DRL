"""Integration tests for custom DQN training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.envs.make_env import make_sysadmin_env
from causal_fmdp_drl.agents.custom_dqn.agent import DQNAgent, DQNConfig
from causal_fmdp_drl.agents.custom_dqn_runner import train_custom_dqn
from causal_fmdp_drl.logging.jsonl_logger import JSONLLogger

DOMAIN_PATH = Path("artifacts/rddl/sysadmin/domain.rddl")


def _read_metrics(output_dir: Path):
    metrics = []
    with open(output_dir / "metrics.jsonl") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def _run_training_inline(tmp_path, lambda_reg, reg_type="none", timesteps=3000):
    """Run training with inline loop for testing regularization types."""
    adj = generate_topology(5, "ring")
    instance_path = write_sysadmin_instance(adj, "test_dqn", tmp_path / "inst", horizon=10)

    env, graph = make_sysadmin_env(DOMAIN_PATH, instance_path, max_episode_steps=10, seed=0)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = DQNConfig(
        lambda_reg=lambda_reg,
        reg_type=reg_type,
        learning_starts=100,
        train_freq=2,
        buffer_size=2000,
        batch_size=16,
        hidden_dim=32,
        eps_decay_steps=1000,
    )
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=config,
        causal_graph=graph if lambda_reg > 0 else None,
    )

    logger = JSONLLogger(output_dir / "metrics.jsonl")

    np.random.seed(0)
    torch.manual_seed(0)
    obs, _ = env.reset(seed=0)
    episode_return = 0.0
    episode_count = 0

    for t in range(timesteps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        losses = agent.train_step(obs, action, reward, next_obs, float(done))
        episode_return += reward

        if done:
            episode_count += 1
            logger.log({"timestep": t, "episode": episode_count, "episode_return": episode_return, **losses})
            obs, _ = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

    torch.save(agent.q_net.state_dict(), output_dir / "q_net.pt")
    logger.close()
    env.close()

    return output_dir, _read_metrics(output_dir)


def _run_training_inline_with_k(tmp_path, lambda_reg, reg_type, k_target, timesteps=3000):
    """Run training with specific k_target for testing gradient_balanced edge cases."""
    adj = generate_topology(5, "ring")
    instance_path = write_sysadmin_instance(adj, "test_dqn", tmp_path / "inst", horizon=10)

    env, graph = make_sysadmin_env(DOMAIN_PATH, instance_path, max_episode_steps=10, seed=0)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = DQNConfig(
        lambda_reg=lambda_reg,
        reg_type=reg_type,
        k_target_override=k_target,
        learning_starts=100,
        train_freq=2,
        buffer_size=2000,
        batch_size=16,
        hidden_dim=32,
        eps_decay_steps=1000,
    )
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=config,
        causal_graph=graph if lambda_reg > 0 else None,
    )

    logger = JSONLLogger(output_dir / "metrics.jsonl")

    np.random.seed(0)
    torch.manual_seed(0)
    obs, _ = env.reset(seed=0)
    episode_return = 0.0
    episode_count = 0

    for t in range(timesteps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        losses = agent.train_step(obs, action, reward, next_obs, float(done))
        episode_return += reward

        if done:
            episode_count += 1
            logger.log({"timestep": t, "episode": episode_count, "episode_return": episode_return, **losses})
            obs, _ = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

    torch.save(agent.q_net.state_dict(), output_dir / "q_net.pt")
    logger.close()
    env.close()

    return output_dir, _read_metrics(output_dir)


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestTrainCustomDQNInline:
    """Tests using inline loop for regularization types."""

    def test_rank_bound_reg_loss_nonzero(self, tmp_path):
        _, metrics = _run_training_inline(tmp_path, lambda_reg=0.01, reg_type="rank_bound")
        assert len(metrics) > 0
        reg_losses = [m["reg_loss"] for m in metrics if "reg_loss" in m]
        assert any(r > 0.0 for r in reg_losses)

    def test_spectral_ratio_reg_loss_nonzero(self, tmp_path):
        _, metrics = _run_training_inline(tmp_path, lambda_reg=0.01, reg_type="spectral_ratio")
        assert len(metrics) > 0
        reg_losses = [m["reg_loss"] for m in metrics if "reg_loss" in m]
        assert any(r > 0.0 for r in reg_losses)

    def test_gradient_balanced_small_k_works(self, tmp_path):
        """Test gradient_balanced with small k_target (normal operation)."""
        _, metrics = _run_training_inline_with_k(
            tmp_path, lambda_reg=0.1, reg_type="gradient_balanced", k_target=4
        )
        assert len(metrics) > 0
        # Should have gate values logged
        gates = [m.get("gate", 0.0) for m in metrics if "gate" in m]
        assert any(g > 0.0 for g in gates), "Gate should be > 0 when tail_ratio > 0"

    def test_gradient_balanced_large_k_no_crash(self, tmp_path):
        """Test gradient_balanced with k_target >= max_rank doesn't crash.

        When k_target >= min(batch_size, hidden_dim), there's no tail to penalize.
        The code should handle this gracefully without attempting to compute
        gradients of a non-differentiable tensor.
        """
        # With batch_size=16, hidden_dim=32, max_rank=16
        # k_target=32 exceeds max_rank, so raw_penalty has requires_grad=False
        _, metrics = _run_training_inline_with_k(
            tmp_path, lambda_reg=0.1, reg_type="gradient_balanced", k_target=32
        )
        assert len(metrics) > 0
        # Should complete without crash; reg_loss should be 0 when k >= max_rank
        # (gate and other diagnostics may be 0 as well)

    def test_gradient_balanced_logs_eff_reg_grad_ratio(self, tmp_path):
        """Test that eff_reg_grad_ratio is logged and approximately equals lambda * gate."""
        _, metrics = _run_training_inline_with_k(
            tmp_path, lambda_reg=0.1, reg_type="gradient_balanced", k_target=4
        )
        # Check that verification metric is logged
        ratios = [m.get("eff_reg_grad_ratio", 0.0) for m in metrics if "eff_reg_grad_ratio" in m]
        gates = [m.get("gate", 0.0) for m in metrics if "gate" in m]
        assert len(ratios) > 0, "eff_reg_grad_ratio should be logged"
        # When gate > 0, eff_reg_grad_ratio should be approximately lambda * gate
        for i, (r, g) in enumerate(zip(ratios, gates)):
            if g > 0.1:  # Only check when gate is meaningfully on
                expected = 0.1 * g  # lambda=0.1
                # Allow 50% tolerance due to numerical issues
                assert abs(r - expected) < 0.5 * expected + 0.01, (
                    f"eff_reg_grad_ratio={r} should be ≈ lambda*gate={expected}"
                )


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestTrainCustomDQNRunner:
    """Tests exercising the train_custom_dqn runner function."""

    @pytest.fixture
    def instance_path(self, tmp_path):
        adj = generate_topology(5, "ring")
        return write_sysadmin_instance(adj, "test_runner", tmp_path / "inst", horizon=10)

    def test_noreg_produces_outputs(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_noreg"
        train_custom_dqn(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            lambda_reg=0.0,
            print_every=0,
        )
        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "graph.json").exists()
        assert (output_dir / "q_net.pt").exists()

    def test_noreg_reg_loss_zero(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_noreg2"
        train_custom_dqn(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            lambda_reg=0.0,
            print_every=0,
        )
        metrics = _read_metrics(output_dir)
        assert len(metrics) > 0
        for m in metrics:
            assert m.get("reg_loss", 0.0) == 0.0

    def test_checkpoint_loadable(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_ckpt"
        train_custom_dqn(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            lambda_reg=0.0,
            print_every=0,
        )
        from causal_fmdp_drl.agents.custom_dqn.network import QNetwork
        net = QNetwork(obs_dim=5, num_actions=6, hidden_dim=DQNConfig().hidden_dim)
        net.load_state_dict(torch.load(output_dir / "q_net.pt", weights_only=True))

    def test_graph_json_has_k_global(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_graph"
        train_custom_dqn(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=1000,
            max_episode_steps=10,
            seed=0,
            lambda_reg=0.0,
            print_every=0,
        )
        with open(output_dir / "graph.json") as f:
            graph_data = json.load(f)
        assert "k_global" in graph_data
        assert graph_data["k_global"] > 0
