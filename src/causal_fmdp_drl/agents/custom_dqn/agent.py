"""Custom DQN agent with optional causal rank regularization."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .replay import ReplayBuffer
from ...regularizers.causal_rank import causal_rank_penalty
from ...graphs.causal_graph import CausalGraph


@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.95
    batch_size: int = 32
    buffer_size: int = 1_000_000
    target_update_freq: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 5_000
    hidden_dim: int = 64
    learning_starts: int = 100
    train_freq: int = 4
    lambda_reg: float = 0.0
    max_grad_norm: float = 10.0
    device: str = "cpu"


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        config: DQNConfig,
        causal_graph: Optional[CausalGraph] = None,
    ):
        self.config = config
        self.num_actions = num_actions

        self.q_net = QNetwork(obs_dim, num_actions, config.hidden_dim).to(config.device)
        self.target_net = QNetwork(obs_dim, num_actions, config.hidden_dim).to(config.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, config.device)

        self.step_count = 0

        self.k_target: Optional[int] = None
        if causal_graph is not None and config.lambda_reg > 0:
            self.k_target = causal_graph.K_causal(num_actions)

    def get_epsilon(self) -> float:
        """Linear epsilon decay."""
        progress = min(1.0, self.step_count / self.config.eps_decay_steps)
        return self.config.eps_start + progress * (self.config.eps_end - self.config.eps_start)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.random() < self.get_epsilon():
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.config.device)
            q_values = self.q_net(obs_t)
            return q_values.argmax(dim=1).item()

    def _update(self) -> dict:
        """Perform one gradient update. Returns loss dict."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config.batch_size)

        q_values, features = self.q_net.forward_with_features(obs)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1)[0]
            targets = rewards + self.config.gamma * next_q * (1 - dones)

        td_loss = nn.functional.smooth_l1_loss(q_values, targets)

        reg_loss = torch.tensor(0.0, device=self.config.device)
        if self.config.lambda_reg > 0 and self.k_target is not None:
            reg_loss = causal_rank_penalty(features, self.k_target)

        total_loss = td_loss + self.config.lambda_reg * reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "td_loss": td_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": total_loss.item(),
            "epsilon": self.get_epsilon(),
        }

    def compute_svd_metrics(self) -> dict:
        """Compute SVD-based representation metrics from a batch of features.

        Returns dict with:
            singular_values: list of singular values (descending)
            effective_rank: Shannon entropy-based effective rank
            rank_above_threshold: count of singular values > 1% of max
        Returns empty dict if buffer has insufficient samples.
        """
        if self.buffer.size < self.config.batch_size:
            return {}

        with torch.no_grad():
            obs_batch, *_ = self.buffer.sample(self.config.batch_size)
            _, features = self.q_net.forward_with_features(obs_batch)
            features_centered = features - features.mean(dim=0, keepdim=True)
            sv = torch.linalg.svdvals(features_centered)

        sv_np = sv.cpu().numpy()

        # Effective rank via Shannon entropy of normalized singular values
        sv_norm = sv_np / (sv_np.sum() + 1e-10)
        sv_norm = sv_norm[sv_norm > 1e-10]  # avoid log(0)
        entropy = -float((sv_norm * np.log(sv_norm)).sum())
        effective_rank = float(np.exp(entropy))

        # Count of singular values above 1% of max
        threshold = 0.01 * sv_np[0] if sv_np[0] > 0 else 0.0
        rank_above = int((sv_np > threshold).sum())

        return {
            "singular_values": sv_np.tolist(),
            "effective_rank": effective_rank,
            "rank_above_threshold": rank_above,
        }

    def train_step(self, obs, action, reward, next_obs, done) -> dict:
        """Add transition to buffer and optionally update."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.step_count += 1

        if self.buffer.size < self.config.learning_starts:
            return {}
        if self.step_count % self.config.train_freq != 0:
            return {}

        return self._update()
