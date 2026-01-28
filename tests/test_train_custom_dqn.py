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


def _run_training_inline(tmp_path, lambda_reg, timesteps=3000):
    """Run training with inline loop (allows k_target override for testing)."""
    adj = generate_topology(5, "ring")
    instance_path = write_sysadmin_instance(adj, "test_dqn", tmp_path / "inst", horizon=10)

    env, graph = make_sysadmin_env(DOMAIN_PATH, instance_path, max_episode_steps=10, seed=0)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = DQNConfig(
        lambda_reg=lambda_reg,
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
    # Override k_target so penalty is nonzero at this scale
    # (K_causal is typically larger than hidden_dim for SysAdmin)
    if lambda_reg > 0:
        agent.k_target = 2

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
    """Tests using inline loop with k_target override."""

    def test_reg_loss_nonzero(self, tmp_path):
        _, metrics = _run_training_inline(tmp_path, lambda_reg=0.01)
        assert len(metrics) > 0
        reg_losses = [m["reg_loss"] for m in metrics if "reg_loss" in m]
        assert any(r > 0.0 for r in reg_losses)


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
        net = QNetwork(obs_dim=5, num_actions=5, hidden_dim=DQNConfig().hidden_dim)
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
