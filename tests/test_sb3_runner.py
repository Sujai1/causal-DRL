"""Tests for SB3 runner."""

import json
from pathlib import Path

import pytest

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.agents.sb3_runner import train_sb3

DOMAIN_PATH = Path("artifacts/rddl/sysadmin/domain.rddl")


@pytest.fixture
def instance_path(tmp_path):
    adj = generate_topology(5, "ring")
    return write_sysadmin_instance(adj, "test_sb3", tmp_path / "instances", horizon=10)


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestTrainSB3:
    def test_ppo_trains(self, instance_path, tmp_path):
        output_dir = tmp_path / "output_ppo"
        train_sb3(
            algo="ppo",
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=2048,
            max_episode_steps=10,
            seed=0,
        )
        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "graph.json").exists()
        assert (output_dir / "ppo_model.zip").exists()

    def test_dqn_trains(self, instance_path, tmp_path):
        output_dir = tmp_path / "output_dqn"
        train_sb3(
            algo="dqn",
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=500,
            max_episode_steps=10,
            seed=0,
            learning_starts=100,
        )
        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "graph.json").exists()

    def test_graph_json_has_k_global(self, instance_path, tmp_path):
        output_dir = tmp_path / "output_graph"
        train_sb3(
            algo="ppo",
            domain_path=DOMAIN_PATH,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=2048,
            max_episode_steps=10,
            seed=0,
        )
        with open(output_dir / "graph.json") as f:
            graph_data = json.load(f)
        assert "k_global" in graph_data
        assert graph_data["k_global"] > 0
