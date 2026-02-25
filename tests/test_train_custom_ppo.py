"""Integration tests for custom PPO training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.agents.custom_ppo.agent import PPOConfig
from causal_fmdp_drl.agents.custom_ppo_runner import train_custom_ppo

DOMAIN_PATH = Path("artifacts/rddl/sysadmin/domain.rddl")


def _read_metrics(output_dir: Path):
    metrics = []
    with open(output_dir / "metrics.jsonl") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestTrainCustomPPO:
    @pytest.fixture
    def instance_path(self, tmp_path):
        adj = generate_topology(5, "ring")
        return write_sysadmin_instance(adj, "test_ppo", tmp_path / "inst", horizon=10)

    def test_produces_outputs(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_ppo"
        train_custom_ppo(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            print_every=0,
            n_steps=128,
        )
        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "graph.json").exists()
        assert (output_dir / "actor.pt").exists()
        assert (output_dir / "critic.pt").exists()

    def test_metrics_have_expected_keys(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_ppo_keys"
        train_custom_ppo(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            print_every=0,
            n_steps=128,
        )
        metrics = _read_metrics(output_dir)
        assert len(metrics) > 0

        # Check all rollout entries have core keys
        for m in metrics:
            assert "timestep" in m
            assert "rollout" in m

        # At least some entries should have PPO metrics
        ppo_entries = [m for m in metrics if "policy_loss" in m]
        assert len(ppo_entries) > 0
        for m in ppo_entries:
            assert "value_loss" in m
            assert "entropy" in m
            assert "clip_fraction" in m
            assert "approx_kl" in m

    def test_graph_json_has_k_global(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_ppo_graph"
        train_custom_ppo(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=1000,
            max_episode_steps=10,
            seed=0,
            print_every=0,
            n_steps=128,
        )
        with open(output_dir / "graph.json") as f:
            graph_data = json.load(f)
        assert "k_global" in graph_data
        assert graph_data["k_global"] > 0

    def test_checkpoint_loadable(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_ppo_ckpt"
        train_custom_ppo(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            print_every=0,
            n_steps=128,
        )
        from causal_fmdp_drl.agents.custom_ppo.network import ActorNetwork, CriticNetwork
        actor = ActorNetwork(obs_dim=5, num_actions=6, hidden_dim=PPOConfig().hidden_dim)
        actor.load_state_dict(torch.load(output_dir / "actor.pt", weights_only=True))

        critic = CriticNetwork(obs_dim=5, hidden_dim=PPOConfig().hidden_dim)
        critic.load_state_dict(torch.load(output_dir / "critic.pt", weights_only=True))

    def test_layernorm_variant(self, instance_path, tmp_path):
        output_dir = tmp_path / "out_ppo_ln"
        train_custom_ppo(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            print_every=0,
            use_layernorm=True,
            n_steps=128,
        )
        metrics = _read_metrics(output_dir)
        assert len(metrics) > 0
        ppo_entries = [m for m in metrics if "policy_loss" in m]
        assert len(ppo_entries) > 0

    def test_multiple_rollouts(self, instance_path, tmp_path):
        """With small n_steps, we should get multiple rollouts."""
        output_dir = tmp_path / "out_ppo_multi"
        train_custom_ppo(
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=3000,
            max_episode_steps=10,
            seed=0,
            print_every=0,
            n_steps=64,
        )
        metrics = _read_metrics(output_dir)
        # 3000 / 64 ≈ 46 rollouts
        assert len(metrics) >= 10, f"Expected multiple rollouts, got {len(metrics)}"
