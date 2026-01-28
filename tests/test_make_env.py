"""Tests for make_sysadmin_env factory."""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.envs.make_env import make_sysadmin_env

DOMAIN_PATH = Path("artifacts/rddl/sysadmin/domain.rddl")


@pytest.fixture
def env_and_graph(tmp_path):
    adj = generate_topology(5, "ring")
    instance_path = write_sysadmin_instance(adj, "test_make", tmp_path, horizon=10)
    env, graph = make_sysadmin_env(DOMAIN_PATH, instance_path, max_episode_steps=10, seed=0)
    yield env, graph
    env.close()


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestMakeSysadminEnv:
    def test_obs_space(self, env_and_graph):
        env, _ = env_and_graph
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (5,)

    def test_action_space(self, env_and_graph):
        env, _ = env_and_graph
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 5

    def test_graph_returned(self, env_and_graph):
        _, graph = env_and_graph
        assert graph.num_vars == 5
        assert graph.k_global > 0

    def test_episode_truncates(self, env_and_graph):
        env, _ = env_and_graph
        obs, _ = env.reset(seed=0)
        steps = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps <= 10

    def test_random_rollout(self, env_and_graph):
        env, _ = env_and_graph
        obs, _ = env.reset(seed=0)
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        assert isinstance(total_reward, float)
