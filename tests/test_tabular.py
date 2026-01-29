"""Tests for tabular RL agents."""

import json
from pathlib import Path

import numpy as np
import pytest

from causal_fmdp_drl.agents.tabular.state_encoding import (
    obs_to_index,
    index_to_obs,
    check_tractable,
)
from causal_fmdp_drl.agents.tabular.q_learning import TabularQLearning, TabularQConfig
from causal_fmdp_drl.agents.tabular.dyna_q import DynaQ, DynaQConfig


class TestStateEncoding:
    def test_obs_to_index_zeros(self):
        obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert obs_to_index(obs) == 0

    def test_obs_to_index_ones(self):
        obs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # binary 111 = 7
        assert obs_to_index(obs) == 7

    def test_obs_to_index_mixed(self):
        obs = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        # binary 0101 (LSB first) = 5
        assert obs_to_index(obs) == 5

    def test_index_to_obs(self):
        obs = index_to_obs(5, 4)
        expected = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)

    def test_roundtrip(self):
        for n in [3, 5, 8]:
            for idx in range(2**n):
                obs = index_to_obs(idx, n)
                assert obs_to_index(obs) == idx

    def test_check_tractable_small(self):
        assert check_tractable(10, max_states=50_000) is True

    def test_check_tractable_large(self, capsys):
        result = check_tractable(20, max_states=50_000)
        assert result is False
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "too large" in captured.out


class TestTabularQLearning:
    def test_initialization(self):
        agent = TabularQLearning(10, 4, TabularQConfig())
        assert agent.Q.shape == (10, 4)
        assert np.all(agent.Q == 0)

    def test_epsilon_starts_at_one(self):
        agent = TabularQLearning(10, 4, TabularQConfig())
        assert agent.get_epsilon() == 1.0

    def test_epsilon_decays(self):
        config = TabularQConfig(eps_decay_steps=100)
        agent = TabularQLearning(10, 4, config)
        agent.step_count = 50
        eps = agent.get_epsilon()
        assert 0.1 < eps < 1.0

    def test_epsilon_reaches_minimum(self):
        config = TabularQConfig(eps_decay_steps=100, eps_end=0.1)
        agent = TabularQLearning(10, 4, config)
        agent.step_count = 200
        assert abs(agent.get_epsilon() - 0.1) < 1e-9

    def test_greedy_action_selection(self):
        agent = TabularQLearning(10, 4, TabularQConfig())
        agent.Q[5, 2] = 10.0  # Make action 2 best for state 5
        action = agent.select_action(5, eval_mode=True)
        assert action == 2

    def test_update_changes_q_value(self):
        agent = TabularQLearning(10, 4, TabularQConfig(lr=0.1, gamma=0.99))
        initial_q = agent.Q[0, 0]
        agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)
        assert agent.Q[0, 0] != initial_q

    def test_update_toward_target(self):
        config = TabularQConfig(lr=1.0, gamma=0.0)  # Full update, no bootstrap
        agent = TabularQLearning(10, 4, config)
        agent.update(state=0, action=0, reward=5.0, next_state=1, done=True)
        assert agent.Q[0, 0] == 5.0

    def test_update_increments_step_count(self):
        agent = TabularQLearning(10, 4, TabularQConfig())
        assert agent.step_count == 0
        agent.update(0, 0, 1.0, 1, False)
        assert agent.step_count == 1


class TestDynaQ:
    def test_inherits_from_q_learning(self):
        agent = DynaQ(10, 4, DynaQConfig())
        assert isinstance(agent, TabularQLearning)

    def test_model_stores_transitions(self):
        agent = DynaQ(10, 4, DynaQConfig(planning_steps=0))
        agent.update(state=0, action=1, reward=1.0, next_state=2, done=False)
        assert (0, 1) in agent.model
        assert agent.model[(0, 1)] == [(1.0, 2, False)]

    def test_seen_pairs_tracked(self):
        agent = DynaQ(10, 4, DynaQConfig(planning_steps=0))
        agent.update(0, 1, 1.0, 2, False)
        agent.update(3, 2, 2.0, 4, True)
        assert (0, 1) in agent.seen_pairs
        assert (3, 2) in agent.seen_pairs

    def test_planning_updates_q_values(self):
        # With planning, Q should change more than without
        config_no_plan = DynaQConfig(planning_steps=0, lr=0.1)
        config_with_plan = DynaQConfig(planning_steps=50, lr=0.1)

        agent_no_plan = DynaQ(10, 4, config_no_plan)
        agent_with_plan = DynaQ(10, 4, config_with_plan)

        # Same transition
        agent_no_plan.update(0, 0, 10.0, 1, False)
        agent_with_plan.update(0, 0, 10.0, 1, False)

        # With planning, more updates happen, Q values should be larger
        # (since we keep updating toward the same positive reward)
        assert agent_with_plan.Q[0, 0] >= agent_no_plan.Q[0, 0]

    def test_planning_uses_stored_transitions(self):
        np.random.seed(42)
        config = DynaQConfig(planning_steps=100, lr=0.5, gamma=0.0)
        agent = DynaQ(10, 4, config)

        # Store one transition with high reward
        agent.update(state=5, action=2, reward=100.0, next_state=6, done=True)

        # After many planning steps, Q(5,2) should be close to 100
        assert agent.Q[5, 2] > 90.0


class TestTabularRunner:
    """Integration tests for tabular runners."""

    @pytest.fixture
    def domain_path(self):
        return Path("artifacts/rddl/sysadmin/domain.rddl")

    @pytest.fixture
    def instance_path(self, tmp_path):
        from causal_fmdp_drl.envs.rddl.instance_generator import (
            generate_topology,
            write_sysadmin_instance,
        )

        adj = generate_topology(3, "ring", seed=0)
        return write_sysadmin_instance(adj, "test_ring_m3", tmp_path)

    def test_tabular_q_produces_outputs(self, domain_path, instance_path, tmp_path):
        from causal_fmdp_drl.agents.tabular_runner import train_tabular_q

        output_dir = tmp_path / "tabular_q_out"
        train_tabular_q(
            domain_path=domain_path,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=500,
            max_episode_steps=20,
            seed=0,
            print_every=0,
        )

        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "graph.json").exists()
        assert (output_dir / "q_table.npy").exists()

        # Check metrics format
        with open(output_dir / "metrics.jsonl") as f:
            first_line = json.loads(f.readline())
        assert "episode_return" in first_line
        assert "epsilon" in first_line

    def test_dyna_q_produces_outputs(self, domain_path, instance_path, tmp_path):
        from causal_fmdp_drl.agents.tabular_runner import train_dyna_q

        output_dir = tmp_path / "dyna_q_out"
        train_dyna_q(
            domain_path=domain_path,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=500,
            max_episode_steps=20,
            seed=0,
            planning_steps=5,
            print_every=0,
        )

        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "graph.json").exists()
        assert (output_dir / "q_table.npy").exists()

    def test_q_table_loadable(self, domain_path, instance_path, tmp_path):
        from causal_fmdp_drl.agents.tabular_runner import train_tabular_q

        output_dir = tmp_path / "tabular_q_out"
        train_tabular_q(
            domain_path=domain_path,
            instance_path=instance_path,
            output_dir=output_dir,
            total_timesteps=200,
            max_episode_steps=20,
            seed=0,
            print_every=0,
        )

        Q = np.load(output_dir / "q_table.npy")
        # 3 machines -> 8 states, 3 actions
        assert Q.shape == (8, 3)
