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


BASELINE_COLORS = {
    "sb3_ppo": "#1f77b4",
    "sb3_dqn": "#ff7f0e",
    "custom_dqn_noreg": "#2ca02c",
    "custom_dqn_reg": "#d62728",
    "tabular_q": "#9467bd",
    "dyna_q": "#8c564b",
}

BASELINE_LABELS = {
    "sb3_ppo": "SB3 PPO",
    "sb3_dqn": "SB3 DQN",
    "custom_dqn_noreg": "Custom DQN (no reg)",
    "custom_dqn_reg": "Custom DQN (reg)",
    "tabular_q": "Tabular Q-Learning",
    "dyna_q": "Dyna-Q",
}


def _color(name: str) -> str:
    return BASELINE_COLORS.get(name, "#333333")


def _label(name: str) -> str:
    return BASELINE_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_learning_curves(
    metrics: dict[str, list[dict]], output_path: Path, smoothing: int = 10
) -> None:
    """Episode return vs timestep for all baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, records in metrics.items():
        timesteps = [r["timestep"] for r in records]
        returns = [r["episode_return"] for r in records]
        smoothed = _rolling_mean(returns, smoothing)
        ax.plot(timesteps, smoothed, label=_label(name), color=_color(name), alpha=0.9)
        ax.plot(timesteps, returns, color=_color(name), alpha=0.15)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for name in ["custom_dqn_noreg", "custom_dqn_reg"]:
        if name not in metrics:
            continue
        records = [r for r in metrics[name] if "td_loss" in r]
        if not records:
            continue
        ax.plot(
            [r["timestep"] for r in records],
            [r["td_loss"] for r in records],
            label=_label(name), color=_color(name),
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
    """Regularization loss vs timestep for custom DQN reg."""
    if "custom_dqn_reg" not in metrics:
        return
    records = [r for r in metrics["custom_dqn_reg"] if "reg_loss" in r]
    if not records:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [r["timestep"] for r in records],
        [r["reg_loss"] for r in records],
        color=_color("custom_dqn_reg"),
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Regularization Loss")
    ax.set_title("Causal Rank Regularization Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_epsilon(metrics: dict[str, list[dict]], output_path: Path) -> None:
    """Epsilon schedule vs timestep for custom DQN variants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for name in ["custom_dqn_noreg", "custom_dqn_reg"]:
        if name not in metrics:
            continue
        records = [r for r in metrics[name] if "epsilon" in r]
        if not records:
            continue
        ax.plot(
            [r["timestep"] for r in records],
            [r["epsilon"] for r in records],
            label=_label(name), color=_color(name),
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
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for name in ["custom_dqn_noreg", "custom_dqn_reg"]:
        if name not in metrics:
            continue
        records = [r for r in metrics[name] if "effective_rank" in r]
        if not records:
            continue
        ax.plot(
            [r["timestep"] for r in records],
            [r["effective_rank"] for r in records],
            label=_label(name), color=_color(name), marker="o", markersize=3,
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
    """Bar chart of singular values at final SVD snapshot, noreg vs reg."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    width = 0.35
    for i, name in enumerate(["custom_dqn_noreg", "custom_dqn_reg"]):
        if name not in metrics:
            continue
        records = [r for r in metrics[name] if "singular_values" in r]
        if not records:
            continue
        sv = np.array(records[-1]["singular_values"])
        x = np.arange(len(sv))
        ax.bar(x + i * width, sv, width, label=_label(name), color=_color(name), alpha=0.8)
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
    names = list(wall_times.keys())
    times = [wall_times[n] for n in names]
    labels = [_label(n) for n in names]
    colors = [_color(n) for n in names]

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
    names = list(stats.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    aucs = [stats[n]["auc"] for n in names]
    colors = [_color(n) for n in names]
    labels = [_label(n) for n in names]
    ax.bar(labels, aucs, color=colors, alpha=0.8)
    ax.set_ylabel("AUC (Return × Timestep)")
    ax.set_title("Sample Efficiency (AUC)")
    ax.grid(True, alpha=0.3, axis="y")

    # Timesteps to threshold
    ax = axes[1]
    ttt = [stats[n]["timesteps_to_threshold"] for n in names]
    ttt_vals = [v if v is not None else 0 for v in ttt]
    bar_colors = [_color(n) if ttt[i] is not None else "#cccccc" for i, n in enumerate(names)]
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
    plot_epsilon(metrics, output_dir / "epsilon_schedule.png")
    plot_effective_rank(metrics, output_dir / "effective_rank.png")
    plot_svd_spectrum(metrics, output_dir / "singular_value_spectrum.png")
    plot_wall_time(run_config, output_dir / "wall_time.png")
    plot_sample_efficiency(metrics, output_dir / "sample_efficiency.png", threshold)
    write_summary(metrics, run_config, output_dir / "summary.json", threshold)

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
