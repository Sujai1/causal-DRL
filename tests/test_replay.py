"""Tests for replay buffer."""

import numpy as np
import torch
import pytest

from causal_fmdp_drl.agents.custom_dqn.replay import ReplayBuffer


class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(capacity=10, obs_dim=4)
        assert buf.size == 0
        buf.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert buf.size == 1

    def test_does_not_exceed_capacity(self):
        buf = ReplayBuffer(capacity=5, obs_dim=2)
        for i in range(20):
            buf.add(np.ones(2) * i, 0, 1.0, np.ones(2) * i, False)
        assert buf.size == 5

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, obs_dim=3)
        for i in range(50):
            buf.add(np.ones(3) * i, i % 4, float(i), np.ones(3) * (i + 1), i == 49)
        obs, actions, rewards, next_obs, dones = buf.sample(16)
        assert obs.shape == (16, 3)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_obs.shape == (16, 3)
        assert dones.shape == (16,)

    def test_sample_returns_tensors(self):
        buf = ReplayBuffer(capacity=10, obs_dim=2)
        for _ in range(10):
            buf.add(np.zeros(2), 0, 0.0, np.zeros(2), False)
        result = buf.sample(4)
        for t in result:
            assert isinstance(t, torch.Tensor)

    def test_sample_dtype(self):
        buf = ReplayBuffer(capacity=10, obs_dim=2)
        for _ in range(10):
            buf.add(np.zeros(2), 1, 0.5, np.ones(2), True)
        obs, actions, rewards, next_obs, dones = buf.sample(4)
        assert obs.dtype == torch.float32
        assert actions.dtype == torch.int64
        assert rewards.dtype == torch.float32
        assert dones.dtype == torch.float32

    def test_wraps_around_correctly(self):
        buf = ReplayBuffer(capacity=3, obs_dim=1)
        buf.add(np.array([10.0]), 0, 0.0, np.array([0.0]), False)
        buf.add(np.array([20.0]), 0, 0.0, np.array([0.0]), False)
        buf.add(np.array([30.0]), 0, 0.0, np.array([0.0]), False)
        buf.add(np.array([40.0]), 0, 0.0, np.array([0.0]), False)  # overwrites index 0
        assert buf.size == 3
        # The oldest entry (10.0) should be gone, 40.0 should be at index 0
        assert buf.obs[0, 0] == 40.0
