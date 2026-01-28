"""Tests for RDDL environment wrappers."""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyRDDLGym
import pytest

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.envs.wrappers import FlattenObsWrapper, SingleRebootActionWrapper

DOMAIN_PATH = Path("artifacts/rddl/sysadmin/domain.rddl")


@pytest.fixture
def raw_env(tmp_path):
    adj = generate_topology(5, "ring")
    instance_path = write_sysadmin_instance(adj, "test_wrap", tmp_path, horizon=10)
    env = pyRDDLGym.make(domain=str(DOMAIN_PATH), instance=str(instance_path))
    yield env
    env.close()


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestFlattenObsWrapper:
    def test_obs_space_is_box(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        assert isinstance(env.observation_space, spaces.Box)

    def test_obs_shape(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        assert env.observation_space.shape == (5,)

    def test_obs_dtype(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_obs_values_are_binary(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        obs, _ = env.reset(seed=0)
        assert all(v in (0.0, 1.0) for v in obs)

    def test_obs_consistent_across_resets(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestSingleRebootActionWrapper:
    def test_action_space_is_discrete(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        env = SingleRebootActionWrapper(env)
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_size(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        env = SingleRebootActionWrapper(env)
        assert env.action_space.n == 5

    def test_step_with_each_action(self, raw_env):
        env = FlattenObsWrapper(raw_env)
        env = SingleRebootActionWrapper(env)
        env.reset(seed=0)
        for a in range(5):
            obs, reward, terminated, truncated, info = env.step(a)
            assert obs.shape == (5,)
            if terminated or truncated:
                env.reset()
