"""Actor and Critic networks with feature extraction hooks for PPO."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class CriticNetwork(nn.Module):
    """MLP state-value network that exposes penultimate layer features.

    Architecture (default):
        obs -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> V(s)
    Architecture (use_layernorm=True):
        obs -> fc1 -> LN -> ReLU -> fc2 -> LN -> ReLU -> fc3 -> V(s)

    The penultimate (fc2) post-ReLU activations are stored for regularization.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        self._features: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar values with shape (batch,)."""
        x = self.fc1(obs)
        if self.use_layernorm:
            x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        if self.use_layernorm:
            x = self.ln2(x)
        x = torch.relu(x)
        self._features = x
        return self.fc3(x).squeeze(-1)

    def get_features(self) -> torch.Tensor:
        """Return features from the last forward pass."""
        if self._features is None:
            raise RuntimeError("Must call forward() before get_features()")
        return self._features

    def forward_with_features(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (values, features) in one call."""
        values = self.forward(obs)
        return values, self._features


class ActorNetwork(nn.Module):
    """MLP policy network that outputs action logits.

    Architecture (default):
        obs -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> logits
    Architecture (use_layernorm=True):
        obs -> fc1 -> LN -> ReLU -> fc2 -> LN -> ReLU -> fc3 -> logits

    The penultimate (fc2) post-ReLU activations are stored for future ablations.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        self._features: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits with shape (batch, num_actions)."""
        x = self.fc1(obs)
        if self.use_layernorm:
            x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        if self.use_layernorm:
            x = self.ln2(x)
        x = torch.relu(x)
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
        """Return (logits, features) in one call."""
        logits = self.forward(obs)
        return logits, self._features

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """Return a Categorical distribution over actions."""
        logits = self.forward(obs)
        return Categorical(logits=logits)
