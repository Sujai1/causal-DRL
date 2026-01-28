"""Q-network with feature extraction hook."""

from typing import Tuple

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP Q-network that exposes penultimate layer features.

    Architecture: obs -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> Q-values
    The penultimate (fc2) activations are stored for regularization.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        self._features: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        self._features = x
        return self.fc3(x)

    def get_features(self) -> torch.Tensor:
        """Return features from the last forward pass."""
        if self._features is None:
            raise RuntimeError("Must call forward() before get_features()")
        return self._features

    def forward_with_features(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q_values, features) in one call."""
        q_values = self.forward(obs)
        return q_values, self._features
