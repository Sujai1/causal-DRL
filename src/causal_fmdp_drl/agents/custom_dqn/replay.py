"""Simple circular replay buffer."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch


@dataclass
class ReplayBuffer:
    capacity: int
    obs_dim: int
    device: str = "cpu"

    def __post_init__(self):
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idxs]).to(self.device),
            torch.from_numpy(self.actions[idxs]).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            torch.from_numpy(self.next_obs[idxs]).to(self.device),
            torch.from_numpy(self.dones[idxs]).to(self.device),
        )
