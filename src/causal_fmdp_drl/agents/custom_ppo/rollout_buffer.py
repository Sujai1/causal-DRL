"""On-policy rollout buffer with truncation-aware GAE."""

import warnings
from typing import Iterator

import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout data and computes GAE with correct truncation handling.

    Stores terminated and truncated flags separately so that:
    - Truncated episodes bootstrap with V(s_{t+1}) (not zero)
    - True terminals get zero bootstrap
    - GAE accumulator resets on any episode boundary (terminated or truncated)
    """

    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device

        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.terminateds = np.zeros(buffer_size, dtype=np.float32)
        self.truncateds = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.ptr

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a single transition."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminateds[self.ptr] = float(terminated)
        self.truncateds[self.ptr] = float(truncated)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_terminated: bool,
        last_truncated: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns with truncation-aware bootstrapping.

        For truncated episodes: bootstrap with V(s_{t+1}) (the value is non-zero).
        For true terminals: no bootstrap (next_value = 0).
        GAE accumulator resets on any episode boundary (terminated or truncated).

        The terminated/truncated flags stored in the buffer are used directly
        for each step. last_terminated/last_truncated are accepted for API
        compatibility but unused (the buffer's stored flags suffice).

        Args:
            last_value: V(s_T) for the state after the last stored transition.
            last_terminated: Unused (kept for API compatibility).
            last_truncated: Unused (kept for API compatibility).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        n = self.size
        last_gae = 0.0

        for t in reversed(range(n)):
            # Determine V(s_{t+1}): use last_value for the final step,
            # otherwise use the stored value at t+1
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # Bootstrap mask: use terminated[t] (not terminated[t+1])
            # If step t is terminated, s_{t+1} is terminal -> no bootstrap
            # If step t is truncated, s_{t+1} is NOT terminal -> bootstrap
            non_terminal = 1.0 - self.terminateds[t]

            # GAE accumulator resets on any done (terminated or truncated)
            done_t = min(self.terminateds[t] + self.truncateds[t], 1.0)

            delta = (
                self.rewards[t]
                + gamma * next_value * non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * (1.0 - done_t) * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_minibatches(self, batch_size: int) -> Iterator[dict]:
        """Yield shuffled minibatches as dicts of tensors.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            Dict with keys: obs, actions, old_log_probs, old_values,
                advantages, returns — all as tensors on self.device.
        """
        n = self.size
        if n == 0:
            return

        if n % batch_size != 0:
            warnings.warn(
                f"buffer_size ({n}) is not a multiple of batch_size ({batch_size}). "
                f"Last {n % batch_size} samples will be dropped.",
                stacklevel=2,
            )

        indices = np.random.permutation(n)

        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.from_numpy(self.obs[idx]).to(self.device),
                "actions": torch.from_numpy(self.actions[idx]).to(self.device),
                "old_log_probs": torch.from_numpy(self.log_probs[idx]).to(self.device),
                "old_values": torch.from_numpy(self.values[idx]).to(self.device),
                "advantages": torch.from_numpy(self.advantages[idx]).to(self.device),
                "returns": torch.from_numpy(self.returns[idx]).to(self.device),
            }

    def reset(self) -> None:
        """Clear buffer for next rollout."""
        self.ptr = 0
        self.full = False
