"""Generate plots from a comparison output directory.

Usage:
    python scripts/plot_results.py outputs/<timestamp>_comparison_m10/
    python scripts/plot_results.py outputs/<dir>/ --threshold 500 --smoothing 10
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(output_dir: Path) -> dict[str, list[dict]]:
    """Load all metrics.jsonl files from subdirectories.

    Returns mapping of baseline name -> list of episode dicts.
    """
    metrics = {}
    for sub in sorted(output_dir.iterdir()):
        jsonl = sub / "metrics.jsonl"
        if sub.is_dir() and jsonl.exists():
            records = []
            for line in jsonl.read_text().strip().splitlines():
                if line.strip():
                    records.append(json.loads(line))
            if records:
                metrics[sub.name] = records
    return metrics


def load_run_config(output_dir: Path) -> dict[str, Any]:
    """Load run_config.json if it exists."""
    path = output_dir / "run_config.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Compute rolling mean with edge handling."""
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < 2:
        return arr
    kernel = np.ones(window) / window
    # Pad to avoid shrinkage
    padded = np.concatenate([np.full(window - 1, arr[0]), arr])
    return np.convolve(padded, kernel, mode="valid")


def _add_jitter(values: np.ndarray, curve_index: int, num_curves: int, jitter_frac: float = 0.02) -> np.ndarray:
    """Add small vertical jitter to separate overlapping curves.

    Args:
        values: Array of y-values
        curve_index: Index of this curve (0, 1, 2, ...)
        num_curves: Total number of curves being plotted
        jitter_frac: Jitter as fraction of data range (default 2%)

    Returns:
        Values with jitter added
    """
    if num_curves <= 1:
        return values
    # Compute jitter offset: spread curves evenly around zero
    data_range = np.ptp(values) if np.ptp(values) > 0 else 1.0
    max_jitter = jitter_frac * data_range
    # Offset from center: -max_jitter to +max_jitter
    offset = (curve_index - (num_curves - 1) / 2) * (2 * max_jitter / max(num_curves - 1, 1))
    return values + offset


BASELINE_COLORS = {
    "sb3_ppo": "#1f77b4",
    "sb3_dqn": "#ff7f0e",
    "custom_dqn_noreg": "#2ca02c",
    "custom_dqn_rank_bound": "#d62728",
    "custom_dqn_spectral_ratio": "#e377c2",
    "tabular_q": "#9467bd",
    "dyna_q": "#8c564b",
}

# Distinct colors for dynamic rank-bound variants (k=2, 3, 4, 5, 6, ...)
RANK_BOUND_COLORS = [
    "#e41a1c",  # red
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#4daf4a",  # green (different from custom_dqn_noreg)
    "#377eb8",  # blue (different from sb3_ppo)
]

# Distinct colors for gradient-balanced variants (different palette)
GRADIENT_BALANCED_COLORS = [
    "#66c2a5",  # teal
    "#fc8d62",  # salmon
    "#8da0cb",  # periwinkle
    "#e78ac3",  # orchid
    "#a6d854",  # yellow-green
    "#ffd92f",  # gold
    "#e5c494",  # tan
    "#b3b3b3",  # silver
]

BASELINE_LABELS = {
    "sb3_ppo": "SB3 PPO",
    "sb3_dqn": "SB3 DQN",
    "custom_dqn_noreg": "Custom DQN (no reg)",
    "custom_dqn_rank_bound": "Custom DQN (Rank-Bound)",
    "custom_dqn_spectral_ratio": "Custom DQN (Spectral-Ratio)",
    "tabular_q": "Tabular Q-Learning",
    "dyna_q": "Dyna-Q",
}


def _get_custom_dqn_variants(metrics: dict) -> tuple[list[str], list[str]]:
    """Get lists of custom DQN variants from available metrics.

    Returns:
        (all_variants, reg_variants) - all custom DQN baselines, and just the regularized ones
    """
    all_variants = []
    reg_variants = []
    for name in metrics.keys():
        if name.startswith("custom_dqn"):
            all_variants.append(name)
            if name != "custom_dqn_noreg":
                reg_variants.append(name)
    return sorted(all_variants), sorted(reg_variants)


def _color(name: str, variant_index: int = 0) -> str:
    if name in BASELINE_COLORS:
        return BASELINE_COLORS[name]
    # Handle dynamic rank-bound variants (custom_dqn_rank_bound_k8, etc.)
    if name.startswith("custom_dqn_rank_bound_k"):
        return RANK_BOUND_COLORS[variant_index % len(RANK_BOUND_COLORS)]
    # Handle dynamic gradient-balanced variants
    if name.startswith("custom_dqn_gradient_balanced_k"):
        return GRADIENT_BALANCED_COLORS[variant_index % len(GRADIENT_BALANCED_COLORS)]
    return "#333333"


def _label(name: str) -> str:
    if name in BASELINE_LABELS:
        return BASELINE_LABELS[name]
    # Handle dynamic rank-bound variants
    if name.startswith("custom_dqn_rank_bound_k"):
        k = name.split("_k")[-1]
        return f"Rank-Bound (k={k})"
    # Handle dynamic gradient-balanced variants
    if name.startswith("custom_dqn_gradient_balanced_k"):
        k = name.split("_k")[-1]
        return f"Grad-Bal (k={k})"
    return name


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _get_variant_color(name: str, rb_idx: int, gb_idx: int) -> tuple[str, int, int]:
    """Get color for a variant and return updated indices."""
    if name.startswith("custom_dqn_gradient_balanced_k"):
        return _color(name, gb_idx), rb_idx, gb_idx + 1
    elif name.startswith("custom_dqn_rank_bound_k"):
        return _color(name, rb_idx), rb_idx + 1, gb_idx
    else:
        return _color(name, 0), rb_idx, gb_idx


def plot_learning_curves(
    metrics: dict[str, list[dict]], output_path: Path, smoothing: int = 10
) -> None:
    """Episode return vs timestep for all baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_names = sorted(metrics.keys())
    num_curves = len(sorted_names)
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(sorted_names):
        records = metrics[name]
        timesteps = [r["timestep"] for r in records]
        returns = [r["episode_return"] for r in records]
        smoothed = _rolling_mean(returns, smoothing)
        smoothed_jittered = _add_jitter(smoothed, curve_idx, num_curves)
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ax.plot(timesteps, smoothed_jittered, label=_label(name), color=color, alpha=0.9)
        ax.plot(timesteps, returns, color=color, alpha=0.15)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Return")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_td_loss(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """TD loss vs timestep for custom DQN variants."""
    all_variants, _ = _get_custom_dqn_variants(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    num_curves = len(all_variants)
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(all_variants):
        records = [r for r in metrics[name] if "td_loss" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        values = np.array([r["td_loss"] for r in records])
        values_jittered = _add_jitter(values, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            values_jittered,
            label=_label(name), color=color,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("TD Loss")
    ax.set_yscale("log")
    ax.set_title("TD Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reg_loss(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Regularization loss vs timestep for custom DQN reg variants."""
    _, reg_variants = _get_custom_dqn_variants(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    num_curves = len(reg_variants)
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(reg_variants):
        records = [r for r in metrics[name] if "reg_loss" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        values = np.array([r["reg_loss"] for r in records])
        values_jittered = _add_jitter(values, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            values_jittered,
            label=_label(name), color=color,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Regularization Loss (raw penalty)")
    ax.set_title("Regularization Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reg_contribution(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Regularization contribution (after loss balancing) vs timestep."""
    _, reg_variants = _get_custom_dqn_variants(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    num_curves = len(reg_variants)

    # Left: reg_contribution absolute value
    ax = axes[0]
    plotted = False
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(reg_variants):
        records = [r for r in metrics[name] if "reg_contribution" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        values = np.array([r["reg_contribution"] for r in records])
        values_jittered = _add_jitter(values, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            values_jittered,
            label=_label(name), color=color,
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reg Contribution (to total loss)")
        ax.set_title("Regularization Contribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Right: ratio of reg_contribution to td_loss
    ax = axes[1]
    plotted_ratio = False
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(reg_variants):
        records = [r for r in metrics[name] if "reg_contribution" in r and "td_loss" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ratios = np.array([r["reg_contribution"] / (r["td_loss"] + 1e-8) for r in records])
        ratios_jittered = _add_jitter(ratios, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            ratios_jittered,
            label=_label(name), color=color,
        )
        plotted_ratio = True
    if plotted_ratio:
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reg / TD Loss Ratio")
        ax.set_title("Regularization Ratio (Loss Balancing Check)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if not plotted and not plotted_ratio:
        plt.close(fig)
        return

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_spread(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Q-value spread (max - min) vs timestep for custom DQN variants."""
    all_variants, _ = _get_custom_dqn_variants(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    num_curves = len(all_variants)
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(all_variants):
        records = [r for r in metrics[name] if "q_spread" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        values = np.array([r["q_spread"] for r in records])
        values_jittered = _add_jitter(values, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            values_jittered,
            label=_label(name), color=color,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Q-Value Spread (max - min)")
    ax.set_title("Q-Value Spread (Action Differentiation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_epsilon(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Epsilon schedule vs timestep for custom DQN variants."""
    all_variants, _ = _get_custom_dqn_variants(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    num_curves = len(all_variants)
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(all_variants):
        records = [r for r in metrics[name] if "epsilon" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        values = np.array([r["epsilon"] for r in records])
        values_jittered = _add_jitter(values, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            values_jittered,
            label=_label(name), color=color,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_effective_rank(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Effective rank vs timestep for custom DQN variants."""
    all_variants, _ = _get_custom_dqn_variants(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    num_curves = len(all_variants)
    rb_idx, gb_idx = 0, 0
    for curve_idx, name in enumerate(all_variants):
        records = [r for r in metrics[name] if "effective_rank" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        values = np.array([r["effective_rank"] for r in records])
        values_jittered = _add_jitter(values, curve_idx, num_curves)
        ax.plot(
            [r["timestep"] for r in records],
            values_jittered,
            label=_label(name), color=color, marker="o", markersize=3,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank of Representation (Shannon Entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_svd_spectrum(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Bar chart of singular values at final SVD snapshot for custom DQN variants."""
    all_variants, _ = _get_custom_dqn_variants(metrics)
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    # Filter to variants that have SVD data
    variants_with_data = [name for name in all_variants
                         if any("singular_values" in r for r in metrics[name])]
    if not variants_with_data:
        plt.close(fig)
        return
    width = 0.8 / len(variants_with_data)
    rb_idx, gb_idx = 0, 0
    for i, name in enumerate(variants_with_data):
        records = [r for r in metrics[name] if "singular_values" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        sv = np.array(records[-1]["singular_values"])
        x = np.arange(len(sv))
        offset = (i - len(variants_with_data) / 2 + 0.5) * width
        ax.bar(x + offset, sv, width, label=_label(name), color=color, alpha=0.8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("Singular Value Spectrum (Final Snapshot)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_wall_time(run_config: dict, output_path: Path) -> None:
    """Bar chart of total wall-clock time per baseline."""
    wall_times = run_config.get("wall_times", {})
    if not wall_times:
        return
    names = sorted(wall_times.keys())
    times = [wall_times[n] for n in names]
    labels = [_label(n) for n in names]
    # Compute colors with proper variant indexing
    colors = []
    rb_idx, gb_idx = 0, 0
    for n in names:
        color, rb_idx, gb_idx = _get_variant_color(n, rb_idx, gb_idx)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, times, color=colors, alpha=0.8)
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Training Wall-Clock Time")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sample_efficiency(
    metrics: dict[str, list[dict]], output_path: Path, threshold: float | None = None
) -> None:
    """AUC bar chart and timesteps-to-threshold."""
    if not metrics:
        return

    # Compute per-baseline stats
    stats = {}
    for name, records in metrics.items():
        returns = [r["episode_return"] for r in records]
        timesteps = [r["timestep"] for r in records]
        if not returns:
            continue
        auc = float(np.trapezoid(returns, timesteps))
        last_10_mean = float(np.mean(returns[-10:])) if len(returns) >= 10 else float(np.mean(returns))
        stats[name] = {"auc": auc, "last_10_mean": last_10_mean, "returns": returns, "timesteps": timesteps}

    if not stats:
        return

    # Auto threshold: 80% of best baseline's final mean
    if threshold is None:
        best_mean = max(s["last_10_mean"] for s in stats.values())
        threshold = 0.8 * best_mean

    # Timesteps to threshold (first episode where rolling mean >= threshold)
    for name, s in stats.items():
        smoothed = _rolling_mean(s["returns"], 10)
        reached = [i for i, v in enumerate(smoothed) if v >= threshold]
        s["timesteps_to_threshold"] = s["timesteps"][reached[0]] if reached else None

    # Plot AUC bar chart
    names = sorted(stats.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    aucs = [stats[n]["auc"] for n in names]
    # Compute colors with proper variant indexing
    colors = []
    rb_idx, gb_idx = 0, 0
    for n in names:
        color, rb_idx, gb_idx = _get_variant_color(n, rb_idx, gb_idx)
        colors.append(color)
    labels = [_label(n) for n in names]
    ax.bar(labels, aucs, color=colors, alpha=0.8)
    ax.set_ylabel("AUC (Return × Timestep)")
    ax.set_title("Sample Efficiency (AUC)")
    ax.grid(True, alpha=0.3, axis="y")

    # Timesteps to threshold
    ax = axes[1]
    ttt = [stats[n]["timesteps_to_threshold"] for n in names]
    ttt_vals = [v if v is not None else 0 for v in ttt]
    bar_colors = [colors[i] if ttt[i] is not None else "#cccccc" for i in range(len(names))]
    bars = ax.bar(labels, ttt_vals, color=bar_colors, alpha=0.8)
    for i, v in enumerate(ttt):
        if v is None:
            bars[i].set_hatch("//")
    ax.set_ylabel("Timesteps")
    ax.set_title(f"Timesteps to Threshold ({threshold:.1f})")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return stats


def plot_gradient_balancing_diagnostics(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Diagnostic plots for gradient-balanced regularization.

    Shows gate, eff_reg_grad_ratio, tail_ratio, and grad_scale for gradient_balanced variants.
    """
    # Filter to gradient_balanced variants that have the diagnostic metrics
    gb_variants = [name for name in sorted(metrics.keys())
                   if name.startswith("custom_dqn_gradient_balanced")]

    if not gb_variants:
        return

    # Check if any variant has the diagnostic metrics
    has_diagnostics = any(
        any("g_td_norm" in r and r.get("g_td_norm", 0) != 0 for r in metrics[name])
        for name in gb_variants
    )
    if not has_diagnostics:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Soft gate value
    ax = axes[0, 0]
    rb_idx, gb_idx = 0, 0
    for name in gb_variants:
        records = [r for r in metrics[name] if "gate" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ax.plot([r["timestep"] for r in records], [r["gate"] for r in records],
                label=_label(name), color=color)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gate Value [0, 1]")
    ax.set_title("Soft Gate (1=full reg, 0=off)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Effective reg grad ratio (verification: should ≈ lambda * gate)
    ax = axes[0, 1]
    rb_idx, gb_idx = 0, 0
    for name in gb_variants:
        records = [r for r in metrics[name] if "eff_reg_grad_ratio" in r and r.get("eff_reg_grad_ratio", 0) > 0]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ax.plot([r["timestep"] for r in records], [r["eff_reg_grad_ratio"] for r in records],
                label=_label(name), color=color)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||∂reg/∂φ|| / ||∂TD/∂φ||")
    ax.set_title("Eff Reg Grad Ratio (should ≈ λ×gate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Tail ratio (relative tail energy)
    ax = axes[0, 2]
    rb_idx, gb_idx = 0, 0
    for name in gb_variants:
        records = [r for r in metrics[name] if "tail_ratio" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ax.plot([r["timestep"] for r in records], [r["tail_ratio"] for r in records],
                label=_label(name), color=color)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Tail Ratio")
    ax.set_title("Relative Tail Energy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Gradient norms (g_td_norm and g_reg_norm)
    ax = axes[1, 0]
    rb_idx, gb_idx = 0, 0
    for name in gb_variants:
        records = [r for r in metrics[name] if "g_td_norm" in r and r.get("g_td_norm", 0) > 0]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        timesteps = [r["timestep"] for r in records]
        ax.plot(timesteps, [r["g_td_norm"] for r in records],
                label=f"{_label(name)} (TD)", color=color, linestyle="-")
        ax.plot(timesteps, [r["g_reg_norm"] for r in records],
                label=f"{_label(name)} (Reg)", color=color, linestyle="--", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms (TD vs Reg)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 5: Gradient scale factor
    ax = axes[1, 1]
    rb_idx, gb_idx = 0, 0
    for name in gb_variants:
        records = [r for r in metrics[name] if "grad_scale" in r and r.get("grad_scale", 0) > 0]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ax.plot([r["timestep"] for r in records], [r["grad_scale"] for r in records],
                label=_label(name), color=color)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Scale Factor (g_td / g_reg)")
    ax.set_title("Gradient Balancing Scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 6: Reg contribution
    ax = axes[1, 2]
    rb_idx, gb_idx = 0, 0
    for name in gb_variants:
        records = [r for r in metrics[name] if "reg_contribution" in r]
        if not records:
            continue
        color, rb_idx, gb_idx = _get_variant_color(name, rb_idx, gb_idx)
        ax.plot([r["timestep"] for r in records], [r["reg_contribution"] for r in records],
                label=_label(name), color=color)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Regularization Contribution")
    ax.set_title("Reg Contribution (Gradient-Balanced)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_auc_rankings(metrics: dict[str, list[dict]]) -> list[dict]:
    """Compute AUC scores and return sorted rankings.

    Returns list of dicts with keys: name, label, auc, final_mean, rank
    """
    rankings = []
    for name, records in metrics.items():
        returns = [r["episode_return"] for r in records]
        timesteps = [r["timestep"] for r in records]
        if not returns:
            continue
        auc = float(np.trapezoid(returns, timesteps))
        final_mean = float(np.mean(returns[-10:])) if len(returns) >= 10 else float(np.mean(returns))
        rankings.append({
            "name": name,
            "label": _label(name),
            "auc": auc,
            "final_mean": final_mean,
        })

    # Sort by AUC descending
    rankings.sort(key=lambda x: x["auc"], reverse=True)

    # Add rank
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    return rankings


def write_auc_rankings(rankings: list[dict], output_path: Path) -> None:
    """Write AUC rankings table to a text file and print to console."""
    if not rankings:
        return

    # Compute column widths
    max_label = max(len(r["label"]) for r in rankings)
    max_label = max(max_label, len("Method"))

    # Build table
    lines = []
    lines.append("=" * 70)
    lines.append("AUC RANKINGS (Higher = Better)")
    lines.append("=" * 70)
    lines.append("")

    # Header
    header = f"{'Rank':<6} {'Method':<{max_label}} {'AUC':>15} {'Final Mean':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    best_auc = rankings[0]["auc"] if rankings else 1
    for r in rankings:
        pct_of_best = 100 * r["auc"] / best_auc if best_auc > 0 else 0
        row = f"{r['rank']:<6} {r['label']:<{max_label}} {r['auc']:>15,.0f} {r['final_mean']:>12.1f}"
        lines.append(row)

    lines.append("")
    lines.append(f"Best AUC: {rankings[0]['label']} ({rankings[0]['auc']:,.0f})")
    lines.append("")

    # Write to file
    content = "\n".join(lines)
    output_path.write_text(content)

    # Print to console
    print()
    print(content)


def write_summary(
    metrics: dict[str, list[dict]], run_config: dict, output_path: Path,
    threshold: float | None = None,
) -> None:
    """Write machine-readable summary.json."""
    summary = {}
    all_last_means = []
    for name, records in metrics.items():
        returns = [r["episode_return"] for r in records]
        timesteps = [r["timestep"] for r in records]
        if not returns:
            continue
        last_10_mean = float(np.mean(returns[-10:])) if len(returns) >= 10 else float(np.mean(returns))
        all_last_means.append(last_10_mean)
        auc = float(np.trapezoid(returns, timesteps))
        summary[name] = {
            "final_mean_return": last_10_mean,
            "auc": auc,
            "num_episodes": len(returns),
        }

    if threshold is None and all_last_means:
        threshold = 0.8 * max(all_last_means)

    for name, records in metrics.items():
        if name not in summary:
            continue
        returns = [r["episode_return"] for r in records]
        timesteps = [r["timestep"] for r in records]
        smoothed = _rolling_mean(returns, 10)
        reached = [i for i, v in enumerate(smoothed) if v >= threshold]
        summary[name]["timesteps_to_threshold"] = timesteps[reached[0]] if reached else None
        summary[name]["threshold"] = threshold

    wall_times = run_config.get("wall_times", {})
    for name in summary:
        if name in wall_times:
            summary[name]["wall_time_seconds"] = wall_times[name]

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots(
    output_dir: Path, smoothing: int = 10, threshold: float | None = None
) -> None:
    """Generate all plots for a comparison directory."""
    metrics = load_metrics(output_dir)
    run_config = load_run_config(output_dir)

    if not metrics:
        print(f"No metrics found in {output_dir}")
        return

    print(f"Found baselines: {', '.join(metrics.keys())}")

    plot_learning_curves(metrics, output_dir / "learning_curves.png", smoothing)
    plot_td_loss(metrics, output_dir / "td_loss.png")
    plot_reg_loss(metrics, output_dir / "reg_loss.png")
    plot_reg_contribution(metrics, output_dir / "reg_contribution.png")
    plot_q_spread(metrics, output_dir / "q_spread.png")
    plot_epsilon(metrics, output_dir / "epsilon_schedule.png")
    plot_effective_rank(metrics, output_dir / "effective_rank.png")
    plot_svd_spectrum(metrics, output_dir / "singular_value_spectrum.png")
    plot_gradient_balancing_diagnostics(metrics, output_dir / "gradient_balancing.png")
    plot_wall_time(run_config, output_dir / "wall_time.png")
    plot_sample_efficiency(metrics, output_dir / "sample_efficiency.png", threshold)
    write_summary(metrics, run_config, output_dir / "summary.json", threshold)

    # Compute and display AUC rankings
    rankings = compute_auc_rankings(metrics)
    write_auc_rankings(rankings, output_dir / "auc_rankings.txt")

    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot results from a comparison run.")
    parser.add_argument("output_dir", type=Path, help="Comparison output directory")
    parser.add_argument("--smoothing", type=int, default=10, help="Rolling mean window")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Return threshold for sample efficiency (default: 80%% of best)")
    args = parser.parse_args()
    generate_all_plots(args.output_dir, args.smoothing, args.threshold)


if __name__ == "__main__":
    main()
