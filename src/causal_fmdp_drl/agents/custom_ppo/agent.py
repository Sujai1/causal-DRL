"""Custom PPO agent with optional critic rank regularization hooks."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .network import ActorNetwork, CriticNetwork
from .rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters with SB3-compatible defaults."""

    # Core PPO
    lr: float = 3e-4
    gamma: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    adv_norm: str = "minibatch"  # "minibatch" | "rollout" | "none"
    separate_optimizers: bool = False

    # Architecture
    hidden_dim: int = 64
    use_layernorm: bool = False

    # Reg hooks (inactive for now)
    lambda_reg: float = 0.0
    reg_type: str = "none"
    k_target_override: Optional[int] = None
    feature_buffer_size: int = 4
    gate_tau: float = 0.005
    reg_warmup_steps: int = 0

    device: str = "cpu"


class PPOAgent:
    """PPO agent with separate actor and critic networks.

    The critic exposes penultimate features for future rank regularization,
    mirroring the DQN agent's structure.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        config: PPOConfig,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        self.actor = ActorNetwork(
            obs_dim, num_actions, config.hidden_dim, config.use_layernorm
        ).to(config.device)
        self.critic = CriticNetwork(
            obs_dim, config.hidden_dim, config.use_layernorm
        ).to(config.device)

        if config.separate_optimizers:
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=config.lr
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=config.lr
            )
            self.optimizer = None
        else:
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            self.optimizer = optim.Adam(params, lr=config.lr)
            self.actor_optimizer = None
            self.critic_optimizer = None

        self.buffer = RolloutBuffer(config.n_steps, obs_dim, config.device)

        self.step_count = 0

        # Regularization state (for future use)
        self.k_target: Optional[int] = config.k_target_override
        self.feature_buffer: list[torch.Tensor] = []

    def select_action(
        self, obs: np.ndarray, eval_mode: bool = False
    ) -> Tuple[int, float, float]:
        """Select action from policy.

        Args:
            obs: Observation array.
            eval_mode: If True, select greedy action (argmax).

        Returns:
            Tuple of (action, log_prob, value).
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.config.device)
            dist = self.actor.get_distribution(obs_t)
            value = self.critic(obs_t).item()

            if eval_mode:
                action = dist.probs.argmax(dim=-1).item()
                log_prob = dist.log_prob(torch.tensor([action])).item()
            else:
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t).item()
                action = action_t.item()

        return action, log_prob, value

    def _compute_explained_variance(
        self, values: np.ndarray, returns: np.ndarray
    ) -> float:
        """Compute explained variance: 1 - Var(returns - values) / Var(returns)."""
        var_returns = float(np.var(returns))
        if var_returns < 1e-8:
            return 0.0
        return float(1.0 - np.var(returns - values) / var_returns)

    def update(self) -> dict:
        """Perform PPO update using data in the rollout buffer.

        Returns:
            Dict of training metrics.
        """
        n = self.buffer.size
        if n == 0:
            return {}

        config = self.config

        # Explained variance before update
        explained_var = self._compute_explained_variance(
            self.buffer.values[: n], self.buffer.returns[: n]
        )

        # Rollout-level advantage normalization
        if config.adv_norm == "rollout":
            adv = self.buffer.advantages[:n]
            adv_mean, adv_std = adv.mean(), adv.std()
            if adv_std > 1e-8:
                self.buffer.advantages[:n] = (adv - adv_mean) / (adv_std + 1e-8)
        else:
            adv = self.buffer.advantages[:n]
            adv_mean, adv_std = adv.mean(), adv.std()

        # Accumulators for logging
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_reg_loss = 0.0
        total_grad_norm = 0.0
        num_updates = 0
        epochs_run = 0

        for epoch in range(config.n_epochs):
            epoch_kl_sum = 0.0
            epoch_updates = 0

            for batch in self.buffer.get_minibatches(config.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                old_values = batch["old_values"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Minibatch-level advantage normalization
                if config.adv_norm == "minibatch":
                    mb_std = advantages.std()
                    if mb_std > 1e-8:
                        advantages = (advantages - advantages.mean()) / (mb_std + 1e-8)

                # --- Policy loss ---
                dist = self.actor.get_distribution(obs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clip fraction for diagnostics
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > config.clip_range).float().mean().item()
                    )

                # --- Value loss ---
                values, features = self.critic.forward_with_features(obs)

                if config.clip_range_vf is not None:
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -config.clip_range_vf,
                        config.clip_range_vf,
                    )
                    vf_loss1 = (values - returns) ** 2
                    vf_loss2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # --- Regularization placeholder ---
                reg_loss = torch.tensor(0.0, device=config.device)
                # Future: extract features for gradient-balanced regularization
                # if config.lambda_reg > 0 and config.reg_type != "none":
                #     reg_loss = compute_reg(features, ...)

                # --- Total loss ---
                loss = (
                    policy_loss
                    + config.vf_coef * value_loss
                    - config.ent_coef * entropy
                    + reg_loss
                )

                # --- Optimizer step ---
                if config.separate_optimizers:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), config.max_grad_norm
                    )
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        self.critic.parameters(), config.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    grad_norm = max(actor_grad_norm.item(), critic_grad_norm.item())
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    all_params = list(self.actor.parameters()) + list(self.critic.parameters())
                    grad_norm_t = nn.utils.clip_grad_norm_(
                        all_params, config.max_grad_norm
                    )
                    self.optimizer.step()
                    grad_norm = grad_norm_t.item()

                # Approximate KL divergence
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction
                total_approx_kl += approx_kl
                total_reg_loss += reg_loss.item()
                total_grad_norm += grad_norm
                num_updates += 1
                epoch_kl_sum += approx_kl
                epoch_updates += 1

            epochs_run = epoch + 1

            # KL early stopping
            if config.target_kl is not None and epoch_updates > 0:
                epoch_mean_kl = epoch_kl_sum / epoch_updates
                if epoch_mean_kl > config.target_kl:
                    break

        if num_updates == 0:
            return {}

        # Compute mean value predictions and returns for diagnostics
        value_pred_mean = float(self.buffer.values[:n].mean())
        return_mean = float(self.buffer.returns[:n].mean())

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "explained_variance": explained_var,
            "adv_mean": float(adv_mean),
            "adv_std": float(adv_std),
            "value_pred_mean": value_pred_mean,
            "return_mean": return_mean,
            "grad_norm": total_grad_norm / num_updates,
            "reg_loss": total_reg_loss / num_updates,
            "epochs_run": epochs_run,
        }

    def compute_svd_metrics(self, obs_batch: Optional[torch.Tensor] = None) -> dict:
        """Compute SVD-based representation metrics on critic features.

        Args:
            obs_batch: Observations to compute features for. If None, uses
                the last rollout buffer's observations.

        Returns:
            Dict with singular_values, effective_rank, rank_above_threshold,
            feature_rank, dims_90pct_energy, dims_95pct_energy.
        """
        if obs_batch is None:
            n = self.buffer.size
            if n < self.config.batch_size:
                return {}
            idx = np.random.choice(n, min(n, 512), replace=False)
            obs_batch = torch.from_numpy(self.buffer.obs[idx]).to(self.config.device)

        with torch.no_grad():
            _, features = self.critic.forward_with_features(obs_batch)
            features_centered = features - features.mean(dim=0, keepdim=True)
            sv = torch.linalg.svdvals(features_centered)

        sv_np = sv.cpu().numpy()
        n_samples = obs_batch.shape[0]

        sv_norm = sv_np / (sv_np.sum() + 1e-10)
        sv_norm = sv_norm[sv_norm > 1e-10]
        entropy = -float((sv_norm * np.log(sv_norm)).sum())
        effective_rank = float(np.exp(entropy))

        threshold = 0.01 * sv_np[0] if sv_np[0] > 0 else 0.0
        rank_above = int((sv_np > threshold).sum())

        sv_scaled = sv_np / np.sqrt(n_samples)
        feature_rank = int(np.sum(sv_scaled > 0.01))

        energy = sv_np ** 2
        total_energy = energy.sum() + 1e-10
        cumulative_energy = np.cumsum(energy) / total_energy
        dims_90 = int(np.searchsorted(cumulative_energy, 0.90)) + 1
        dims_95 = int(np.searchsorted(cumulative_energy, 0.95)) + 1

        return {
            "singular_values": sv_np.tolist(),
            "effective_rank": effective_rank,
            "rank_above_threshold": rank_above,
            "feature_rank": feature_rank,
            "dims_90pct_energy": dims_90,
            "dims_95pct_energy": dims_95,
        }

    def compute_collapse_diagnostics(self) -> dict:
        """Compute rank collapse diagnostics from rollout buffer.

        Returns dict with dead_feature_ratio, feature_std_median,
        rank_mean, rank_std. Returns empty dict if buffer has insufficient samples.
        """
        n = self.buffer.size
        sample_size = min(512, n)
        if sample_size < self.config.batch_size:
            return {}

        device = self.config.device
        hidden_dim = self.config.hidden_dim

        with torch.no_grad():
            idx = np.random.choice(n, sample_size, replace=False)
            obs_large = torch.from_numpy(self.buffer.obs[idx]).to(device)
            _, features_large = self.critic.forward_with_features(obs_large)

            per_unit_std = features_large.std(dim=0)
            dead_count = int((per_unit_std < 1e-5).sum().item())
            dead_feature_ratio = dead_count / hidden_dim
            feature_std_median = float(per_unit_std.median().item())

            num_batches = 5
            ranks = []
            for _ in range(num_batches):
                batch_idx = np.random.choice(n, min(self.config.batch_size, n), replace=False)
                obs_batch = torch.from_numpy(self.buffer.obs[batch_idx]).to(device)
                _, feats = self.critic.forward_with_features(obs_batch)
                feats_centered = feats - feats.mean(dim=0, keepdim=True)
                sv = torch.linalg.svdvals(feats_centered)
                sv_np = sv.cpu().numpy()
                sv_norm = sv_np / (sv_np.sum() + 1e-10)
                sv_norm = sv_norm[sv_norm > 1e-10]
                entropy = -float((sv_norm * np.log(sv_norm)).sum())
                ranks.append(float(np.exp(entropy)))

        return {
            "dead_feature_ratio": dead_feature_ratio,
            "feature_std_median": feature_std_median,
            "rank_mean": float(np.mean(ranks)),
            "rank_std": float(np.std(ranks)),
        }
