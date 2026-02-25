"""Tests for PPO rollout buffer with truncation-aware GAE."""

import numpy as np
import pytest
import torch

from causal_fmdp_drl.agents.custom_ppo.rollout_buffer import RolloutBuffer


class TestRolloutBufferBasic:
    def test_add_and_size(self):
        buf = RolloutBuffer(buffer_size=10, obs_dim=4)
        assert buf.size == 0
        buf.add(np.zeros(4), 0, 1.0, False, False, -0.5, 0.3)
        assert buf.size == 1
        buf.add(np.zeros(4), 1, 2.0, False, False, -0.3, 0.4)
        assert buf.size == 2

    def test_full_flag(self):
        buf = RolloutBuffer(buffer_size=3, obs_dim=2)
        for i in range(3):
            buf.add(np.zeros(2), 0, 1.0, False, False, 0.0, 0.0)
        assert buf.full
        assert buf.size == 3

    def test_reset_clears_buffer(self):
        buf = RolloutBuffer(buffer_size=5, obs_dim=2)
        for _ in range(5):
            buf.add(np.zeros(2), 0, 1.0, False, False, 0.0, 0.0)
        assert buf.size == 5
        buf.reset()
        assert buf.size == 0
        assert not buf.full


class TestGAEComputation:
    def test_simple_no_done(self):
        """GAE for a short trajectory with no termination.

        Trajectory: r=[1, 1, 1], V=[0.5, 0.5, 0.5], last_value=0.5, gamma=0.99, lam=0.95
        No dones, so GAE should accumulate backwards.
        """
        buf = RolloutBuffer(buffer_size=3, obs_dim=1)
        for i in range(3):
            buf.add(np.array([float(i)]), 0, 1.0, False, False, 0.0, 0.5)

        gamma, lam = 0.99, 0.95
        buf.compute_returns_and_advantages(
            last_value=0.5, last_terminated=False, last_truncated=False,
            gamma=gamma, gae_lambda=lam,
        )

        # delta_t = 1 + 0.99*0.5 - 0.5 = 0.995 for all t
        delta = 1.0 + gamma * 0.5 - 0.5
        # A[2] = delta
        # A[1] = delta + gamma*lam*A[2]
        # A[0] = delta + gamma*lam*A[1]
        expected_a2 = delta
        expected_a1 = delta + gamma * lam * expected_a2
        expected_a0 = delta + gamma * lam * expected_a1

        assert buf.advantages[2] == pytest.approx(expected_a2, abs=1e-5)
        assert buf.advantages[1] == pytest.approx(expected_a1, abs=1e-5)
        assert buf.advantages[0] == pytest.approx(expected_a0, abs=1e-5)

        # Returns = advantages + values
        assert buf.returns[0] == pytest.approx(expected_a0 + 0.5, abs=1e-5)

    def test_terminated_resets_gae(self):
        """Terminated episodes should zero out the bootstrap and reset GAE accumulator.

        Trajectory: t=0: r=1, terminated=True; t=1: r=1, not done
        Values: [0.5, 0.5], last_value=0.5
        """
        buf = RolloutBuffer(buffer_size=2, obs_dim=1)
        buf.add(np.array([0.0]), 0, 1.0, True, False, 0.0, 0.5)   # terminated
        buf.add(np.array([1.0]), 0, 1.0, False, False, 0.0, 0.5)   # not done

        gamma, lam = 0.99, 0.95
        buf.compute_returns_and_advantages(
            last_value=0.5, last_terminated=False, last_truncated=False,
            gamma=gamma, gae_lambda=lam,
        )

        # t=0: terminated, so bootstrap is masked: delta = r - V = 1 - 0.5 = 0.5
        # done=1 resets GAE accumulator, so A[0] = 0.5
        assert buf.advantages[0] == pytest.approx(0.5, abs=1e-5)

        # t=1: not done, delta = 1 + 0.99*0.5 - 0.5 = 0.995
        # A[1] = 0.995 (last step, no future GAE to propagate)
        delta_1 = 1.0 + gamma * 0.5 - 0.5
        assert buf.advantages[1] == pytest.approx(delta_1, abs=1e-5)

    def test_truncated_bootstraps_value(self):
        """Truncated episodes should bootstrap with V(s_{t+1}), not zero.

        Trajectory: t=0: r=1, truncated=True; t=1: r=1, not done
        The key difference from terminated: truncated still bootstraps.
        """
        buf = RolloutBuffer(buffer_size=2, obs_dim=1)
        buf.add(np.array([0.0]), 0, 1.0, False, True, 0.0, 0.5)   # truncated
        buf.add(np.array([1.0]), 0, 1.0, False, False, 0.0, 0.5)   # not done

        gamma, lam = 0.99, 0.95
        buf.compute_returns_and_advantages(
            last_value=0.5, last_terminated=False, last_truncated=False,
            gamma=gamma, gae_lambda=lam,
        )

        # t=1: delta = 1 + 0.99*0.5 - 0.5 = 0.995; A[1] = 0.995
        delta_1 = 1.0 + gamma * 0.5 - 0.5

        # t=0: truncated (not terminated), so non_terminal = 1-terminateds[0] = 1-0 = 1
        # delta = 1 + 0.99*V[1]*1 - 0.5 = 1 + 0.99*0.5 - 0.5 = 0.995
        # done[0] = truncated[0] = 1, so GAE resets:
        # last_gae = 0.995 + 0.99*0.95*(1-1)*A[1] = 0.995
        delta_0 = 1.0 + gamma * 0.5 * 1.0 - 0.5  # bootstraps!
        assert buf.advantages[0] == pytest.approx(delta_0, abs=1e-5)
        assert buf.advantages[1] == pytest.approx(delta_1, abs=1e-5)

        # Key: truncated advantage > terminated advantage because it bootstraps
        assert buf.advantages[0] > 0.5  # terminated case gives 0.5

    def test_returns_equal_advantages_plus_values(self):
        buf = RolloutBuffer(buffer_size=5, obs_dim=2)
        for i in range(5):
            buf.add(np.zeros(2), 0, float(i), False, False, 0.0, float(i) * 0.1)

        buf.compute_returns_and_advantages(
            last_value=1.0, last_terminated=False, last_truncated=False,
            gamma=0.99, gae_lambda=0.95,
        )

        for i in range(5):
            expected_return = buf.advantages[i] + buf.values[i]
            assert buf.returns[i] == pytest.approx(expected_return, abs=1e-5)


class TestMinibatches:
    def test_minibatch_coverage(self):
        """All indices should be covered exactly once per call."""
        buf = RolloutBuffer(buffer_size=64, obs_dim=4)
        for i in range(64):
            buf.add(np.full(4, float(i)), i % 3, 1.0, False, False, 0.0, 0.0)

        buf.compute_returns_and_advantages(0.0, False, False, 0.99, 0.95)

        all_obs = []
        for batch in buf.get_minibatches(16):
            assert batch["obs"].shape == (16, 4)
            assert batch["actions"].shape == (16,)
            assert batch["advantages"].shape == (16,)
            all_obs.append(batch["obs"])

        combined = torch.cat(all_obs, dim=0)
        assert combined.shape == (64, 4)
        # Each observation's first element is unique (0..63)
        unique_vals = set(combined[:, 0].numpy().tolist())
        assert len(unique_vals) == 64

    def test_minibatch_warns_on_remainder(self):
        buf = RolloutBuffer(buffer_size=10, obs_dim=2)
        for _ in range(10):
            buf.add(np.zeros(2), 0, 1.0, False, False, 0.0, 0.0)

        buf.compute_returns_and_advantages(0.0, False, False, 0.99, 0.95)

        with pytest.warns(UserWarning, match="not a multiple"):
            batches = list(buf.get_minibatches(3))
        # 10 // 3 = 3 batches, last sample dropped
        assert len(batches) == 3

    def test_empty_buffer_yields_nothing(self):
        buf = RolloutBuffer(buffer_size=10, obs_dim=2)
        batches = list(buf.get_minibatches(5))
        assert len(batches) == 0
