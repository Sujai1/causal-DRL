"""Custom DQN agent with optional causal rank regularization."""

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .replay import ReplayBuffer
from ...regularizers.causal_rank import (
    causal_rank_penalty,
    nuclear_norm_penalty,
    relative_tail_energy,
)
from ...graphs.causal_graph import CausalGraph


@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.95
    batch_size: int = 32
    buffer_size: int = 1_000_000
    target_update_freq: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 5_000
    hidden_dim: int = 64
    learning_starts: int = 100
    train_freq: int = 4
    lambda_reg: float = 0.0
    reg_type: str = "none"  # "none", "rank_bound", "spectral_ratio", "gradient_balanced", "infer", "gradient_balanced_infer"
    k_target_override: Optional[int] = None  # Manual k_target for rank_bound (default: use k_global)
    feature_buffer_size: int = 4  # Number of past batches to accumulate for gradient_balanced SVD
    gate_tau: float = 0.005  # Soft gate threshold for gradient_balanced (0.5% tail energy)
    reg_warmup_steps: int = 0  # Steps before regularization starts (0 = no warmup, use learning_starts)
    max_grad_norm: float = 10.0
    # InFeR hyperparameters (for reg_type="infer" or "gradient_balanced_infer")
    infer_k: int = 8  # Projection dimension for auxiliary heads
    infer_beta: float = 1.0  # Amplification factor for init targets
    infer_alpha: float = 0.01  # Gradient-balanced λ for InFeR (target ratio of InFeR grad to TD grad)
    use_layernorm: bool = True  # Add LayerNorm before each ReLU activation (prevents rank collapse)
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

        self.q_net = QNetwork(obs_dim, num_actions, config.hidden_dim, config.use_layernorm).to(config.device)
        self.target_net = QNetwork(obs_dim, num_actions, config.hidden_dim, config.use_layernorm).to(config.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, config.device)

        self.step_count = 0

        self.k_target: Optional[int] = None
        self.sparsity_ratio: float = 1.0

        # Feature buffer for gradient_balanced mode (stores detached past features)
        self.feature_buffer: list[torch.Tensor] = []

        # InFeR: frozen init encoder + random projection
        self.init_net: Optional[QNetwork] = None
        self.infer_W: Optional[nn.Linear] = None
        if config.reg_type in ("infer", "gradient_balanced_infer"):
            # Frozen copy of the initial encoder
            self.init_net = copy.deepcopy(self.q_net)
            self.init_net.eval()
            for p in self.init_net.parameters():
                p.requires_grad_(False)
            # Frozen random projection: hidden_dim -> infer_k
            self.infer_W = nn.Linear(config.hidden_dim, config.infer_k, bias=False).to(config.device)
            self.infer_W.requires_grad_(False)

        if causal_graph is not None and config.lambda_reg > 0:
            if config.reg_type in ("rank_bound", "gradient_balanced", "gradient_balanced_infer"):
                if config.k_target_override is not None:
                    self.k_target = config.k_target_override
                else:
                    self.k_target = causal_graph.k_global
            elif config.reg_type == "spectral_ratio":
                max_k = causal_graph.num_vars - 1
                self.sparsity_ratio = 1.0 - (causal_graph.k_global / max_k) if max_k > 0 else 0.0

    def get_epsilon(self) -> float:
        """Linear epsilon decay."""
        progress = min(1.0, self.step_count / self.config.eps_decay_steps)
        return self.config.eps_start + progress * (self.config.eps_end - self.config.eps_start)

    def _get_buffered_features(self, current_features: torch.Tensor) -> torch.Tensor:
        """Concatenate past (detached) features with current (gradient-enabled) features.

        This provides a larger effective batch for stable SVD estimates while
        ensuring gradients only flow through the current batch's features.

        Args:
            current_features: Features from current batch (requires_grad=True)

        Returns:
            Concatenated features: [past_detached..., current_with_grad]
        """
        # Concatenate: past features (detached) + current features (with gradients)
        all_features = self.feature_buffer + [current_features]
        combined = torch.cat(all_features, dim=0)

        # Update buffer with detached current features for next iteration
        self.feature_buffer.append(current_features.detach())
        if len(self.feature_buffer) > self.config.feature_buffer_size:
            self.feature_buffer.pop(0)

        return combined

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

        # Initialize metrics (as tensors where appropriate)
        reg_loss = torch.tensor(0.0, device=self.config.device)
        reg_contribution = torch.tensor(0.0, device=self.config.device)
        # Diagnostic values (will be converted to Python floats for logging)
        g_td_norm_val = 0.0
        g_reg_norm_val = 0.0
        grad_scale_val = 0.0
        tail_ratio_val = 0.0
        gate_val = 0.0
        eff_reg_grad_ratio = 0.0  # Verification: should ≈ lambda * gate when working
        infer_grad_scale_val = 0.0  # Gradient scale for InFeR (when gradient-balanced)

        if self.config.lambda_reg > 0:
            target_ratio = self.config.lambda_reg

            if self.config.reg_type == "rank_bound" and self.k_target is not None:
                # Original value-based balancing
                raw_penalty = causal_rank_penalty(features, self.k_target)
                # Check if penalty is non-trivial (avoid tensor boolean issue)
                if raw_penalty.item() > 1e-10:
                    scale = (td_loss / (raw_penalty + 1e-8)).detach().clamp(0.01, 100.0)
                    reg_contribution = target_ratio * scale * raw_penalty
                reg_loss = raw_penalty

            elif self.config.reg_type == "gradient_balanced" and self.k_target is not None:
                # Gradient-balanced mode with relative tail energy and soft gate

                # Warmup: skip regularization during initial learning phase
                # This prevents early regularization from shaping the representation
                # before the network has learned meaningful features
                # Default warmup = eps_decay_steps (typically 10% of training)
                warmup_steps = self.config.reg_warmup_steps
                if warmup_steps == 0:
                    warmup_steps = self.config.eps_decay_steps  # Default: match epsilon decay
                in_warmup = self.step_count < warmup_steps

                # Get buffered features for stable SVD (larger effective batch)
                buffered_features = self._get_buffered_features(features)

                # Relative tail energy (scale-invariant, kept as tensor)
                raw_penalty = relative_tail_energy(buffered_features, self.k_target)
                reg_loss = raw_penalty
                tail_ratio_val = raw_penalty.item() if raw_penalty.requires_grad else 0.0

                # Skip regularization during warmup or if k_target >= max_rank
                if raw_penalty.requires_grad and not in_warmup:
                    # Soft gate: smoothly turns off when constraint is satisfied
                    # gate ≈ 1 when tail_ratio >> tau, gate ≈ 0 when tail_ratio << tau
                    tau = self.config.gate_tau
                    gate = (raw_penalty / (raw_penalty + tau)).detach()
                    gate_val = gate.item()

                    # If gate is essentially off, skip regularization entirely
                    # This ensures grad-bal with slack k behaves identically to no-reg
                    if gate_val >= 0.001:  # Only apply regularization if gate is meaningful
                        # Gradient balancing: compute gradient norms as tensors
                        g_td_tuple = torch.autograd.grad(
                            td_loss, features, retain_graph=True, allow_unused=True
                        )
                        g_reg_tuple = torch.autograd.grad(
                            raw_penalty, features, retain_graph=True, allow_unused=True
                        )

                        g_td = g_td_tuple[0]
                        g_reg = g_reg_tuple[0]

                        if g_td is not None and g_reg is not None:
                            # Keep as tensors for clean computation
                            g_td_norm = torch.linalg.vector_norm(g_td)
                            g_reg_norm = torch.linalg.vector_norm(g_reg)

                            # Compute scale with cap, keep as tensor
                            scale = torch.clamp(
                                g_td_norm / (g_reg_norm + 1e-8), max=1000.0
                            ).detach()

                            # Apply soft gate to regularization contribution
                            reg_contribution = gate * target_ratio * scale * raw_penalty

                            # Compute verification metric: effective reg gradient magnitude
                            # eff_reg_grad_norm ≈ gate * lambda * g_td_norm (when balancing works)
                            eff_reg_grad_norm = gate * target_ratio * scale * g_reg_norm

                            # Log values (convert to Python floats only here)
                            g_td_norm_val = g_td_norm.item()
                            g_reg_norm_val = g_reg_norm.item()
                            grad_scale_val = scale.item()
                            gate_val = gate.item()
                            tail_ratio_val = raw_penalty.item()
                            # Verification: this should ≈ lambda * gate
                            eff_reg_grad_ratio = (eff_reg_grad_norm / (g_td_norm + 1e-8)).item()
                        else:
                            # Gradient flow is broken - log warning via metrics
                            g_td_norm_val = -1.0
                            g_reg_norm_val = -1.0
                            tail_ratio_val = raw_penalty.item()
                    # else: gate < 0.001, skip regularization entirely (behaves like no-reg)
                # else: k_target >= max_rank or in warmup, reg_contribution stays 0

            elif self.config.reg_type == "spectral_ratio":
                raw_penalty = nuclear_norm_penalty(features)
                if raw_penalty.item() > 1e-10:
                    scale = (td_loss / (raw_penalty + 1e-8)).detach().clamp(0.01, 100.0)
                    reg_contribution = target_ratio * self.sparsity_ratio * scale * raw_penalty
                reg_loss = raw_penalty

        # InFeR: compute InFeR loss for reg_types that use it
        infer_loss_val = 0.0
        if self.config.reg_type in ("infer", "gradient_balanced_infer") and self.infer_W is not None:
            with torch.no_grad():
                _, init_features = self.init_net.forward_with_features(obs)
            z = self.infer_W(features)                          # (batch, k) — has grad
            z0 = self.infer_W(init_features)                    # (batch, k) — no grad
            infer_penalty = ((z - self.config.infer_beta * z0) ** 2).mean()
            infer_loss_val = infer_penalty.item()

            if self.config.reg_type == "infer":
                # Gradient-balanced InFeR: scale InFeR gradients to match TD magnitude
                # infer_alpha acts as λ (target ratio of InFeR grad to TD grad)
                # No gate (InFeR should always be active), no warmup (prevent early collapse)
                reg_loss = infer_penalty
                g_td_tuple = torch.autograd.grad(
                    td_loss, features, retain_graph=True, allow_unused=True
                )
                g_infer_tuple = torch.autograd.grad(
                    infer_penalty, features, retain_graph=True, allow_unused=True
                )
                g_td = g_td_tuple[0]
                g_infer = g_infer_tuple[0]
                if g_td is not None and g_infer is not None:
                    g_td_norm = torch.linalg.vector_norm(g_td)
                    g_infer_norm = torch.linalg.vector_norm(g_infer)
                    scale = torch.clamp(
                        g_td_norm / (g_infer_norm + 1e-8), max=1000.0
                    ).detach()
                    reg_contribution = self.config.infer_alpha * scale * infer_penalty

                    g_td_norm_val = g_td_norm.item()
                    g_reg_norm_val = g_infer_norm.item()
                    grad_scale_val = scale.item()
                    eff_reg_grad_norm = self.config.infer_alpha * scale * g_infer_norm
                    eff_reg_grad_ratio = (eff_reg_grad_norm / (g_td_norm + 1e-8)).item()

            elif self.config.reg_type == "gradient_balanced_infer" and self.k_target is not None:
                # 3-term loss: TD + gb(tail_energy) + gb(InFeR)
                #
                # Tail energy: gradient-balanced with gate/warmup (same as "gradient_balanced")
                # InFeR: independently gradient-balanced, always active from step 0
                target_ratio = self.config.lambda_reg

                # --- Tail energy (gradient-balanced, with warmup + gate) ---
                warmup_steps = self.config.reg_warmup_steps
                if warmup_steps == 0:
                    warmup_steps = self.config.eps_decay_steps
                in_warmup = self.step_count < warmup_steps

                buffered_features = self._get_buffered_features(features)
                tail_penalty = relative_tail_energy(buffered_features, self.k_target)
                reg_loss = tail_penalty  # log the tail energy as reg_loss
                tail_ratio_val = tail_penalty.item() if tail_penalty.requires_grad else 0.0

                if tail_penalty.requires_grad and not in_warmup:
                    tau = self.config.gate_tau
                    gate = (tail_penalty / (tail_penalty + tau)).detach()
                    gate_val = gate.item()

                    if gate_val >= 0.001:
                        g_td_tuple = torch.autograd.grad(
                            td_loss, features, retain_graph=True, allow_unused=True
                        )
                        g_reg_tuple = torch.autograd.grad(
                            tail_penalty, features, retain_graph=True, allow_unused=True
                        )
                        g_td = g_td_tuple[0]
                        g_reg = g_reg_tuple[0]

                        if g_td is not None and g_reg is not None:
                            g_td_norm = torch.linalg.vector_norm(g_td)
                            g_reg_norm = torch.linalg.vector_norm(g_reg)
                            scale = torch.clamp(
                                g_td_norm / (g_reg_norm + 1e-8), max=1000.0
                            ).detach()
                            reg_contribution = gate * target_ratio * scale * tail_penalty

                            eff_reg_grad_norm = gate * target_ratio * scale * g_reg_norm
                            g_td_norm_val = g_td_norm.item()
                            g_reg_norm_val = g_reg_norm.item()
                            grad_scale_val = scale.item()
                            gate_val = gate.item()
                            tail_ratio_val = tail_penalty.item()
                            eff_reg_grad_ratio = (eff_reg_grad_norm / (g_td_norm + 1e-8)).item()

                # --- InFeR: gradient-balanced, always active, no warmup ---
                # Independently scale InFeR gradients to match TD magnitude
                g_td_for_infer = torch.autograd.grad(
                    td_loss, features, retain_graph=True, allow_unused=True
                )[0]
                g_infer = torch.autograd.grad(
                    infer_penalty, features, retain_graph=True, allow_unused=True
                )[0]
                if g_td_for_infer is not None and g_infer is not None:
                    g_td_infer_norm = torch.linalg.vector_norm(g_td_for_infer)
                    g_infer_norm = torch.linalg.vector_norm(g_infer)
                    infer_scale = torch.clamp(
                        g_td_infer_norm / (g_infer_norm + 1e-8), max=1000.0
                    ).detach()
                    infer_contribution = self.config.infer_alpha * infer_scale * infer_penalty
                    reg_contribution = reg_contribution + infer_contribution
                    infer_grad_scale_val = infer_scale.item()

        total_loss = td_loss + reg_contribution

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Compute Q-value spread (max - min) to detect compression
        with torch.no_grad():
            q_all, _ = self.q_net.forward_with_features(obs)
            q_spread = (q_all.max(dim=1)[0] - q_all.min(dim=1)[0]).mean().item()

        return {
            "td_loss": td_loss.item(),
            "reg_loss": reg_loss.item(),  # Raw penalty (for monitoring)
            "reg_contribution": reg_contribution.item(),  # Actual loss contribution
            "total_loss": total_loss.item(),
            "q_spread": q_spread,
            "epsilon": self.get_epsilon(),
            # Gradient balancing diagnostics (for tail energy in gradient_balanced modes)
            "g_td_norm": g_td_norm_val,
            "g_reg_norm": g_reg_norm_val,
            "grad_scale": grad_scale_val,
            "tail_ratio": tail_ratio_val,
            "gate": gate_val,  # Soft gate value [0, 1]
            "eff_reg_grad_ratio": eff_reg_grad_ratio,  # Should ≈ lambda * gate
            "infer_loss": infer_loss_val,
            "infer_grad_scale": infer_grad_scale_val,
        }

    def compute_collapse_diagnostics(self, probe_steps: int = 200) -> dict:
        """Compute rank collapse diagnostics from replay buffer samples.

        Returns dict with:
            dead_feature_ratio: fraction of hidden units with std < 1e-5
            feature_std_median: median per-unit standard deviation
            rank_mean: effective rank averaged over multiple independent batches
            rank_std: std of effective rank across batches
            probe_mse: MSE of linear probe reconstructing obs from frozen features
        Returns empty dict if buffer has insufficient samples.
        """
        sample_size = min(512, self.buffer.size)
        if sample_size < self.config.batch_size:
            return {}

        device = self.config.device
        hidden_dim = self.config.hidden_dim

        with torch.no_grad():
            # Sample a large batch for feature variance analysis
            obs_large, *_ = self.buffer.sample(sample_size)
            _, features_large = self.q_net.forward_with_features(obs_large)

            # --- (a) Dead feature ratio ---
            per_unit_std = features_large.std(dim=0)  # (hidden_dim,)
            dead_count = int((per_unit_std < 1e-5).sum().item())
            dead_feature_ratio = dead_count / hidden_dim
            feature_std_median = float(per_unit_std.median().item())

            # --- (b) Multi-batch rank consistency ---
            num_batches = 5
            ranks = []
            for _ in range(num_batches):
                obs_batch, *_ = self.buffer.sample(self.config.batch_size)
                _, feats = self.q_net.forward_with_features(obs_batch)
                feats_centered = feats - feats.mean(dim=0, keepdim=True)
                sv = torch.linalg.svdvals(feats_centered)
                sv_np = sv.cpu().numpy()
                sv_norm = sv_np / (sv_np.sum() + 1e-10)
                sv_norm = sv_norm[sv_norm > 1e-10]
                entropy = -float((sv_norm * np.log(sv_norm)).sum())
                ranks.append(float(np.exp(entropy)))
            rank_mean = float(np.mean(ranks))
            rank_std = float(np.std(ranks))

        # --- (c) Linear probe accuracy ---
        # Train a small linear layer to reconstruct obs from frozen features
        probe = nn.Linear(hidden_dim, obs_large.shape[1]).to(device)
        probe_opt = optim.Adam(probe.parameters(), lr=1e-3)

        for _ in range(probe_steps):
            obs_batch, *_ = self.buffer.sample(self.config.batch_size)
            with torch.no_grad():
                _, feats = self.q_net.forward_with_features(obs_batch)
            pred = probe(feats)
            loss = nn.functional.mse_loss(pred, obs_batch)
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()

        # Final probe MSE on a held-out sample
        with torch.no_grad():
            obs_eval, *_ = self.buffer.sample(self.config.batch_size)
            _, feats_eval = self.q_net.forward_with_features(obs_eval)
            probe_mse = float(nn.functional.mse_loss(probe(feats_eval), obs_eval).item())

        del probe, probe_opt  # clean up

        return {
            "dead_feature_ratio": dead_feature_ratio,
            "feature_std_median": feature_std_median,
            "rank_mean": rank_mean,
            "rank_std": rank_std,
            "probe_mse": probe_mse,
        }

    def compute_svd_metrics(self) -> dict:
        """Compute SVD-based representation metrics from a batch of features.

        All metrics use the same centered feature matrix for consistency.

        Returns dict with:
            singular_values: list of singular values (descending)
            effective_rank: Shannon entropy-based effective rank
            rank_above_threshold: count of singular values > 1% of max
            feature_rank: numerical feature rank (Lyle et al. 2022) with
                1/sqrt(n) scaling and absolute threshold eps=0.01
            dims_90pct_energy: min dimensions explaining 90% of cumulative energy
            dims_95pct_energy: min dimensions explaining 95% of cumulative energy
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
        n = obs_batch.shape[0]

        # Effective rank via Shannon entropy of normalized singular values
        sv_norm = sv_np / (sv_np.sum() + 1e-10)
        sv_norm = sv_norm[sv_norm > 1e-10]  # avoid log(0)
        entropy = -float((sv_norm * np.log(sv_norm)).sum())
        effective_rank = float(np.exp(entropy))

        # Count of singular values above 1% of max
        threshold = 0.01 * sv_np[0] if sv_np[0] > 0 else 0.0
        rank_above = int((sv_np > threshold).sum())

        # Numerical feature rank (Lyle et al. 2022): count SVs of
        # (1/sqrt(n)) * centered_features that exceed absolute threshold eps.
        # No normalization by max SV — detects uniform shrinkage to zero.
        sv_scaled = sv_np / np.sqrt(n)
        feature_rank = int(np.sum(sv_scaled > 0.01))

        # Cumulative energy dimensions: min m such that
        # sum_{i=1}^{m} sigma_i^2 / sum_j sigma_j^2 >= threshold.
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

    def train_step(self, obs, action, reward, next_obs, done) -> dict:
        """Add transition to buffer and optionally update."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.step_count += 1

        if self.buffer.size < self.config.learning_starts:
            return {}
        if self.step_count % self.config.train_freq != 0:
            return {}

        return self._update()
