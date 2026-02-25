"""Analyze results across multiple seeds from the seed sweep.

Scans all matching experiments, computes cross-seed statistics,
and generates tables + plots.

Usage:
    python scripts/analyze_seed_sweep.py
    python scripts/analyze_seed_sweep.py --outputs_dir outputs/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from plot_results import (
    BASELINE_COLORS,
    BASELINE_LABELS,
    GRADIENT_BALANCED_COLORS,
    _color,
    _label,
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _matches_seed_sweep_config(run_config: dict, ba_m: int | None = None) -> bool:
    """Check if a run_config matches the seed sweep parameters."""
    if not (
        run_config.get("topology") == "barabasi_albert"
        and run_config.get("num_machines") == 10
        and run_config.get("timesteps") == 200000
    ):
        return False
    if ba_m is not None:
        return run_config.get("ba_m") == ba_m
    return True


def compute_graph_stats(graph_data: dict) -> dict:
    """Compute degree distribution statistics from graph.json adjacency."""
    adj = np.array(graph_data["adjacency"])
    n = adj.shape[0]
    # Degree = number of neighbors (exclude self-loops on diagonal)
    degrees = adj.sum(axis=1) - np.diag(adj)

    sorted_deg = np.sort(degrees)[::-1]
    gini = _gini_coefficient(degrees)

    # Hub-related metrics
    hub_mask = degrees >= 5
    hub_edges = adj[hub_mask].sum() - np.diag(adj)[hub_mask].sum()
    total_edges = adj.sum() - np.trace(adj)
    hub_edge_frac = float(hub_edges / total_edges) if total_edges > 0 else 0.0

    sorted_deg = np.sort(degrees)[::-1]
    deg_sum = degrees.sum()
    top2_share = float(sorted_deg[:2].sum() / deg_sum) if deg_sum > 0 else 0.0
    top3_share = float(sorted_deg[:3].sum() / deg_sum) if deg_sum > 0 else 0.0

    hub_ratio = float(degrees.max() / degrees.mean()) if degrees.mean() > 0 else 0.0
    deg_range = float(degrees.max() - degrees.min())

    # Degree entropy (higher = more uniform)
    p = degrees / deg_sum if deg_sum > 0 else np.zeros_like(degrees)
    deg_entropy = float(-np.sum(p * np.log(p + 1e-10)))

    return {
        "n_nodes": n,
        "degrees": degrees.tolist(),
        "deg_min": float(degrees.min()),
        "deg_max": float(degrees.max()),
        "deg_mean": float(degrees.mean()),
        "deg_std": float(degrees.std()),
        "deg_gini": gini,
        "deg_range": deg_range,
        "deg_entropy": deg_entropy,
        "num_hubs": int(np.sum(hub_mask)),
        "num_low_deg": int(np.sum(degrees <= 2)),
        "hub_ratio": hub_ratio,
        "hub_edge_frac": hub_edge_frac,
        "top2_deg_share": top2_share,
        "top3_deg_share": top3_share,
        "density": float(total_edges) / (n * (n - 1)),
    }


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for an array of values."""
    vals = np.sort(values)
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * vals) - (n + 1) * vals.sum()) / (n * vals.sum()))


def _load_metrics_jsonl(path: Path) -> list[dict]:
    """Load a metrics.jsonl file into a list of record dicts."""
    records = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _auc_at_cutoff(records: list[dict], cutoff: int) -> float | None:
    """Compute AUC from raw episode records up to a timestep cutoff."""
    filtered = [
        (r["timestep"], r["episode_return"])
        for r in records if r["timestep"] <= cutoff
    ]
    if len(filtered) < 2:
        return None
    ts, rets = zip(*filtered)
    return float(np.trapezoid(rets, ts))


def _recompute_summary_at_cutoff(exp_dir: Path, cutoff: int) -> dict:
    """Recompute summary AUC values from raw metrics at a given cutoff."""
    summary = {}
    for sub in sorted(exp_dir.iterdir()):
        jsonl = sub / "metrics.jsonl"
        if not (sub.is_dir() and jsonl.exists()):
            continue
        records = _load_metrics_jsonl(jsonl)
        if not records:
            continue
        auc = _auc_at_cutoff(records, cutoff)
        if auc is not None:
            # Also compute final_mean from episodes up to cutoff
            filtered_returns = [
                r["episode_return"] for r in records if r["timestep"] <= cutoff
            ]
            final_mean = float(np.mean(filtered_returns[-10:])) if len(filtered_returns) >= 10 else float(np.mean(filtered_returns))
            summary[sub.name] = {
                "auc": auc,
                "final_mean_return": final_mean,
                "num_episodes": len(filtered_returns),
            }
    return summary


def collect_experiments(
    outputs_dir: Path,
    ba_m: int | None = None,
    cutoff: int | None = None,
) -> list[dict]:
    """Scan output directories and collect per-seed data.

    Args:
        outputs_dir: Root outputs directory.
        ba_m: Filter to this ba_m value, or None for all.
        cutoff: If set, recompute AUC from raw metrics up to this
                timestep instead of using summary.json.

    Returns list of dicts, one per seed, with keys:
        seed, dir_name, summary, graph_stats, run_config
    """
    experiments = []
    seen_seeds = {}

    for exp_dir in sorted(outputs_dir.iterdir()):
        if "_m10" not in exp_dir.name:
            continue
        rc_path = exp_dir / "run_config.json"
        summary_path = exp_dir / "summary.json"
        graph_path = exp_dir / "graph.json"

        if not (rc_path.exists() and summary_path.exists() and graph_path.exists()):
            continue

        run_config = json.loads(rc_path.read_text())
        if not _matches_seed_sweep_config(run_config, ba_m=ba_m):
            continue

        seed = run_config["seed"]

        if cutoff is not None:
            summary = _recompute_summary_at_cutoff(exp_dir, cutoff)
        else:
            summary = json.loads(summary_path.read_text())

        graph_data = json.loads(graph_path.read_text())
        graph_stats = compute_graph_stats(graph_data)

        entry = {
            "seed": seed,
            "dir_name": exp_dir.name,
            "summary": summary,
            "graph_stats": graph_stats,
            "run_config": run_config,
        }

        # Keep only the latest run per seed (sorted dirs → last one wins)
        seen_seeds[seed] = entry

    experiments = sorted(seen_seeds.values(), key=lambda e: e["seed"])
    return experiments


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def _get_auc(summary: dict, baseline: str) -> float | None:
    """Get AUC for a baseline from summary, or None if missing."""
    entry = summary.get(baseline)
    if entry is None:
        return None
    return entry.get("auc")


def _get_final_mean(summary: dict, baseline: str) -> float | None:
    entry = summary.get(baseline)
    if entry is None:
        return None
    return entry.get("final_mean_return")


def _struct_aware_gap_pct(summary: dict) -> float | None:
    """Compute structure-aware heuristic gap % over random_down.

    Gap = (max(highest_deg, most_down_nbrs) AUC - random_down AUC) / |random_down AUC| * 100
    """
    random_down_auc = _get_auc(summary, "heuristic_random_down")
    highest_deg_auc = _get_auc(summary, "heuristic_highest_degree")
    most_down_auc = _get_auc(summary, "heuristic_most_down_neighbors")

    if random_down_auc is None:
        return None
    struct_aware_auc = max(
        v for v in [highest_deg_auc, most_down_auc] if v is not None
    )
    if abs(random_down_auc) < 1e-9:
        return None
    return (struct_aware_auc - random_down_auc) / abs(random_down_auc) * 100


def _gb_vs_ln_pct(summary: dict, gb_key: str) -> float | None:
    """Compute grad-bal variant's AUC gap % over DQN+LN."""
    ln_auc = _get_auc(summary, "custom_dqn_noreg_ln")
    gb_auc = _get_auc(summary, gb_key)
    if ln_auc is None or gb_auc is None or abs(ln_auc) < 1e-9:
        return None
    return (gb_auc - ln_auc) / abs(ln_auc) * 100


def _winner_for_k(summary: dict, gb_key: str) -> str:
    """Determine winner between a specific GB variant and DQN+LN based on AUC."""
    ln_auc = _get_auc(summary, "custom_dqn_noreg_ln")
    gb_auc = _get_auc(summary, gb_key)
    if ln_auc is None or gb_auc is None:
        return "N/A"
    return "GB" if gb_auc > ln_auc else "LN"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_per_seed_table(experiments: list[dict], output_dir: Path) -> None:
    """Print and save per-seed breakdown table."""
    lines = []
    lines.append("=" * 110)
    lines.append("PER-SEED BREAKDOWN")
    lines.append("=" * 110)
    lines.append("")

    header = (
        f"{'Seed':>4}  {'deg_std':>7}  {'struct%':>7}  "
        f"{'k8_vs_LN%':>9}  {'k8_win':>6}  "
        f"{'k16_vs_LN%':>10}  {'k16_win':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for exp in experiments:
        s = exp["summary"]
        gs = exp["graph_stats"]
        struct_gap = _struct_aware_gap_pct(s)
        k8_gap = _gb_vs_ln_pct(s, "custom_dqn_gradient_balanced_k8")
        k16_gap = _gb_vs_ln_pct(s, "custom_dqn_gradient_balanced_k16")
        k8_win = _winner_for_k(s, "custom_dqn_gradient_balanced_k8")
        k16_win = _winner_for_k(s, "custom_dqn_gradient_balanced_k16")

        def fmt(v):
            return f"{v:>7.1f}%" if v is not None else "    N/A"

        lines.append(
            f"{exp['seed']:>4}  {gs['deg_std']:>7.2f}  "
            f"{fmt(struct_gap)}  {fmt(k8_gap)}  {k8_win:>6}  "
            f"{fmt(k16_gap):>10}  {k16_win:>7}"
        )

    lines.append("")
    content = "\n".join(lines)
    print(content)
    (output_dir / "per_seed_table.txt").write_text(content)


def print_grouped_summary(experiments: list[dict], output_dir: Path) -> None:
    """Print and save grouped averages table."""
    # Compute per-seed metrics
    records = []
    for exp in experiments:
        s = exp["summary"]
        struct_gap = _struct_aware_gap_pct(s)
        k8_gap = _gb_vs_ln_pct(s, "custom_dqn_gradient_balanced_k8")
        k16_gap = _gb_vs_ln_pct(s, "custom_dqn_gradient_balanced_k16")
        k8_win = _winner_for_k(s, "custom_dqn_gradient_balanced_k8")
        k16_win = _winner_for_k(s, "custom_dqn_gradient_balanced_k16")
        if struct_gap is not None and (k8_gap is not None or k16_gap is not None):
            records.append({
                "struct_gap": struct_gap,
                "k8_gap": k8_gap,
                "k16_gap": k16_gap,
                "k8_win": k8_win,
                "k16_win": k16_win,
            })

    # Define groups
    high_struct = [r for r in records if r["struct_gap"] > 10]
    low_struct = [r for r in records if r["struct_gap"] <= 10]

    groups = [
        ("struct_gap > 10%", high_struct),
        ("struct_gap <= 10%", low_struct),
        ("All seeds", records),
    ]

    lines = []
    lines.append("=" * 100)
    lines.append("GROUPED SUMMARY")
    lines.append("=" * 100)
    lines.append("")

    header = (
        f"{'Group':<20}  {'n':>3}  {'k8_vs_LN%':>12}  {'k8 win%':>8}  "
        f"{'k16_vs_LN%':>12}  {'k16 win%':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for label, group in groups:
        n = len(group)
        if n == 0:
            lines.append(
                f"{label:<20}  {0:>3}  {'N/A':>12}  {'N/A':>8}  "
                f"{'N/A':>12}  {'N/A':>9}"
            )
            continue

        def mean_std(key):
            vals = [r[key] for r in group if r[key] is not None]
            if not vals:
                return "N/A"
            return f"{np.mean(vals):+.1f}±{np.std(vals):.1f}"

        k8_wr = sum(1 for r in group if r["k8_win"] == "GB") / n * 100
        k16_wr = sum(1 for r in group if r["k16_win"] == "GB") / n * 100

        lines.append(
            f"{label:<20}  {n:>3}  {mean_std('k8_gap'):>12}  {k8_wr:>7.0f}%  "
            f"{mean_std('k16_gap'):>12}  {k16_wr:>8.0f}%"
        )

    lines.append("")
    content = "\n".join(lines)
    print(content)
    (output_dir / "grouped_summary.txt").write_text(content)


def print_graph_properties(experiments: list[dict], output_dir: Path) -> None:
    """Print and save graph properties table, grouped by winner."""
    lines = []
    lines.append("=" * 100)
    lines.append("GRAPH PROPERTIES BY SEED")
    lines.append("=" * 100)
    lines.append("")

    header = (
        f"{'Seed':>4}  {'k8_win':>6}  {'k16_win':>7}  {'deg_min':>7}  {'deg_max':>7}  "
        f"{'deg_mean':>8}  {'deg_std':>7}  {'#hubs≥5':>7}  {'#low≤2':>6}  "
        f"{'density':>7}  {'Gini':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    k8_gb_stats = []
    k8_ln_stats = []
    k16_gb_stats = []
    k16_ln_stats = []

    for exp in experiments:
        gs = exp["graph_stats"]
        k8_win = _winner_for_k(exp["summary"], "custom_dqn_gradient_balanced_k8")
        k16_win = _winner_for_k(exp["summary"], "custom_dqn_gradient_balanced_k16")
        lines.append(
            f"{exp['seed']:>4}  {k8_win:>6}  {k16_win:>7}  {gs['deg_min']:>7.1f}  "
            f"{gs['deg_max']:>7.1f}  {gs['deg_mean']:>8.2f}  "
            f"{gs['deg_std']:>7.2f}  {gs['num_hubs']:>7}  "
            f"{gs['num_low_deg']:>6}  {gs['density']:>7.3f}  "
            f"{gs['deg_gini']:>5.3f}"
        )
        (k8_gb_stats if k8_win == "GB" else k8_ln_stats).append(gs)
        (k16_gb_stats if k16_win == "GB" else k16_ln_stats).append(gs)

    # Grouped means
    lines.append("")
    lines.append("-" * len(header))

    for label, bucket in [
        ("k8: GB wins", k8_gb_stats), ("k8: LN wins", k8_ln_stats),
        ("k16: GB wins", k16_gb_stats), ("k16: LN wins", k16_ln_stats),
    ]:
        n = len(bucket)
        if n == 0:
            lines.append(f"{label} (n=0): no data")
            continue
        lines.append(
            f"{label} (n={n}):  "
            f"deg_mean={np.mean([g['deg_mean'] for g in bucket]):.2f}  "
            f"deg_std={np.mean([g['deg_std'] for g in bucket]):.2f}  "
            f"#hubs={np.mean([g['num_hubs'] for g in bucket]):.1f}  "
            f"#low={np.mean([g['num_low_deg'] for g in bucket]):.1f}  "
            f"density={np.mean([g['density'] for g in bucket]):.3f}  "
            f"Gini={np.mean([g['deg_gini'] for g in bucket]):.3f}"
        )

    lines.append("")
    content = "\n".join(lines)
    print(content)
    (output_dir / "graph_properties.txt").write_text(content)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_struct_gap_vs_gb_advantage(
    experiments: list[dict], output_dir: Path,
) -> None:
    """Scatter: structure-aware gap % vs k=8 and k=16 advantage % over DQN+LN."""
    GB_K8 = "custom_dqn_gradient_balanced_k8"
    GB_K16 = "custom_dqn_gradient_balanced_k16"

    variants = [
        (GB_K8, "k=8", "#66c2a5", "o"),
        (GB_K16, "k=16", "#fc8d62", "s"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    all_x, all_y = [], []  # for axis limits

    for gb_key, vlabel, color, marker in variants:
        x_vals, y_vals, seed_labels = [], [], []
        for exp in experiments:
            sg = _struct_aware_gap_pct(exp["summary"])
            gap = _gb_vs_ln_pct(exp["summary"], gb_key)
            if sg is not None and gap is not None:
                x_vals.append(sg)
                y_vals.append(gap)
                seed_labels.append(str(exp["seed"]))

        if not x_vals:
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)
        all_x.extend(x_vals)
        all_y.extend(y_vals)

        ax.scatter(x, y, s=60, zorder=5, color=color, edgecolors="white",
                   linewidth=0.5, marker=marker, label=vlabel)

        # Label each point with seed number
        for xi, yi, lbl in zip(x, y, seed_labels):
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="#555555")

        # Regression line + Spearman per variant
        if len(x) >= 3:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(min(all_x) - 1, max(all_x) + 1, 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), "--", color=color, alpha=0.5)

            rho, pval = spearmanr(x, y)
            ax.annotate(
                f"{vlabel}: ρ={rho:.3f}, p={pval:.3f}",
                xy=(0.05, 0.95 if gb_key == GB_K8 else 0.89),
                xycoords="axes fraction", verticalalignment="top", fontsize=9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
            )

    # Reference lines
    ax.axhline(0, color="red", linestyle=":", alpha=0.5, label="Break-even (GB = LN)")
    ax.axvline(10, color="green", linestyle=":", alpha=0.5, label="Struct gap = 10%")

    ax.set_xlabel("Structure-Aware Heuristic Gap % (over Random Down)")
    ax.set_ylabel("Grad-Bal AUC Gap % (over DQN+LN)")
    ax.set_title("Grad-Bal Advantage vs Structure-Aware Heuristic Gap (k=8 & k=16)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "struct_gap_vs_gb_advantage.png", dpi=150)
    plt.close(fig)
    print(f"  Saved struct_gap_vs_gb_advantage.png")


def plot_graph_property_vs_gb_advantage(
    experiments: list[dict], output_dir: Path,
) -> None:
    """Multi-panel scatter: various graph properties vs k=8 and k=16 advantage % over DQN+LN."""
    GB_K8 = "custom_dqn_gradient_balanced_k8"
    GB_K16 = "custom_dqn_gradient_balanced_k16"

    # (graph_stats key, x-axis label, is_derived)
    x_axis_specs = [
        # Hub-centric metrics
        ("num_hubs", "# Hubs (deg >= 5)", False),
        ("hub_ratio", "Hub Ratio (max_deg / mean_deg)", False),
        ("hub_edge_frac", "Hub Edge Fraction", False),
        ("top2_deg_share", "Top-2 Nodes Degree Share", False),
        ("top3_deg_share", "Top-3 Nodes Degree Share", False),
        # Degree distribution shape
        ("deg_std", "Degree Std Dev", False),
        ("deg_range", "Degree Range (max - min)", False),
        ("deg_gini", "Degree Gini Coefficient", False),
        ("deg_entropy", "Degree Entropy (higher=uniform)", False),
        # Extremes
        ("deg_max", "Max Degree", False),
        ("deg_min", "Min Degree", False),
        ("num_low_deg", "# Low-Degree Nodes (deg <= 2)", False),
        # Derived
        ("struct_gap", "Structure-Aware Heuristic Gap %", True),
    ]

    n_panels = len(x_axis_specs)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols

    variants = [
        (GB_K8, "k=8", "#66c2a5", "o"),
        (GB_K16, "k=16", "#fc8d62", "s"),
    ]

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = axes.flatten()

    for panel_idx, (key, xlabel, is_derived) in enumerate(x_axis_specs):
        ax = axes_flat[panel_idx]

        for gb_key, vlabel, color, marker in variants:
            x_vals, y_vals = [], []
            for exp in experiments:
                gap = _gb_vs_ln_pct(exp["summary"], gb_key)
                if gap is None:
                    continue
                if is_derived:
                    xval = _struct_aware_gap_pct(exp["summary"])
                else:
                    xval = exp["graph_stats"].get(key)
                if xval is None:
                    continue
                x_vals.append(xval)
                y_vals.append(gap)

            if len(x_vals) < 3:
                continue

            x = np.array(x_vals)
            y = np.array(y_vals)

            ax.scatter(x, y, s=40, zorder=5, color=color, edgecolors="white",
                       linewidth=0.4, marker=marker, label=vlabel)

            # Regression line
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, np.polyval(coeffs, x_line), "--", color=color, alpha=0.5)

            # Spearman
            rho, pval = spearmanr(x, y)
            y_pos = 0.95 if gb_key == GB_K8 else 0.85
            star = "*" if pval < 0.05 else ""
            ax.annotate(
                f"{vlabel}: ρ={rho:.2f}{star}",
                xy=(0.03, y_pos), xycoords="axes fraction",
                verticalalignment="top", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.5),
            )

        ax.axhline(0, color="red", linestyle=":", alpha=0.4)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("GB AUC Gap % over LN", fontsize=9)
        ax.set_title(f"GB Advantage vs {xlabel}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused panels
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Grad-Bal Advantage vs Graph Properties (k=8 & k=16)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "graph_properties_vs_gb_advantage.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved graph_properties_vs_gb_advantage.png")


def plot_avg_auc_by_method(experiments: list[dict], output_dir: Path) -> None:
    """Bar chart: average AUC by method across all seeds."""
    # Collect AUCs per baseline across all experiments
    auc_by_baseline: dict[str, list[float]] = {}
    for exp in experiments:
        for baseline, data in exp["summary"].items():
            auc = data.get("auc")
            if auc is not None:
                auc_by_baseline.setdefault(baseline, []).append(auc)

    if not auc_by_baseline:
        print("No AUC data for bar chart.")
        return

    # Sort by mean AUC descending
    sorted_baselines = sorted(
        auc_by_baseline.keys(),
        key=lambda b: np.mean(auc_by_baseline[b]),
        reverse=True,
    )

    means = [np.mean(auc_by_baseline[b]) for b in sorted_baselines]
    stds = [np.std(auc_by_baseline[b]) for b in sorted_baselines]
    labels = [_label(b) for b in sorted_baselines]
    colors = [_color(b) for b in sorted_baselines]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_baselines))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUC (across seeds)")
    ax.set_title(f"Average AUC by Method (n={len(experiments)} seeds)")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate count
    for i, b in enumerate(sorted_baselines):
        n = len(auc_by_baseline[b])
        if n < len(experiments):
            ax.annotate(f"n={n}", (i, means[i] + stds[i]),
                        ha="center", va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(output_dir / "avg_auc_by_method.png", dpi=150)
    plt.close(fig)
    print(f"  Saved avg_auc_by_method.png")


def plot_gb_advantage_histogram(experiments: list[dict], output_dir: Path) -> None:
    """Side-by-side histograms of k=8 and k=16 vs DQN+LN % gap across all seeds."""
    GB_K8 = "custom_dqn_gradient_balanced_k8"
    GB_K16 = "custom_dqn_gradient_balanced_k16"

    k8_gaps = [g for g in (_gb_vs_ln_pct(e["summary"], GB_K8) for e in experiments) if g is not None]
    k16_gaps = [g for g in (_gb_vs_ln_pct(e["summary"], GB_K16) for e in experiments) if g is not None]

    if not k8_gaps and not k16_gaps:
        print("No data for GB advantage histogram.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, gaps, vlabel, color in [
        (axes[0], k8_gaps, "k=8", "#66c2a5"),
        (axes[1], k16_gaps, "k=16", "#fc8d62"),
    ]:
        if not gaps:
            ax.set_title(f"Grad-Bal ({vlabel}) — no data")
            continue
        gaps_arr = np.array(gaps)
        n_wins = int(np.sum(gaps_arr > 0))
        n_total = len(gaps_arr)

        ax.hist(gaps_arr, bins=12, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
        ax.axvline(np.mean(gaps_arr), color="blue", linestyle="--", linewidth=1,
                   label=f"Mean = {np.mean(gaps_arr):+.2f}%")
        ax.axvline(np.median(gaps_arr), color="orange", linestyle="--", linewidth=1,
                   label=f"Median = {np.median(gaps_arr):+.2f}%")

        ax.annotate(
            f"GB wins: {n_wins}/{n_total} ({n_wins/n_total*100:.0f}%)",
            xy=(0.95, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

        ax.set_xlabel(f"Grad-Bal ({vlabel}) AUC Gap % over DQN+LN")
        ax.set_ylabel("Number of Seeds")
        ax.set_title(f"Grad-Bal ({vlabel}) Advantage Distribution")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "gb_advantage_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved gb_advantage_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-seed results from the seed sweep."
    )
    parser.add_argument(
        "--outputs_dir", type=Path, default=Path("outputs"),
        help="Root outputs directory (default: outputs/)",
    )
    parser.add_argument(
        "--ba_m", type=int, default=None,
        help="Filter to experiments with this ba_m value (default: all ba_m values)",
    )
    parser.add_argument(
        "--cutoff", type=int, default=None,
        help="Recompute AUC from raw metrics up to this timestep "
             "(e.g. 100000 for AUC@100k). Default: use full-run summary.json.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated list of seeds to include, or a range like 3-24. "
             "Default: all seeds.",
    )
    args = parser.parse_args()

    # Parse --seeds filter
    seed_filter = None
    if args.seeds is not None:
        if "-" in args.seeds and "," not in args.seeds:
            lo, hi = args.seeds.split("-")
            seed_filter = set(range(int(lo), int(hi) + 1))
        else:
            seed_filter = set(int(s) for s in args.seeds.split(","))

    ba_m_label = f"ba_m={args.ba_m}" if args.ba_m is not None else "all ba_m"
    cutoff_label = f", AUC@{args.cutoff//1000}k" if args.cutoff else ""
    seed_label = f", seeds={args.seeds}" if args.seeds else ""
    print(f"Collecting seed sweep experiments ({ba_m_label}{cutoff_label}{seed_label})...")
    experiments = collect_experiments(args.outputs_dir, ba_m=args.ba_m, cutoff=args.cutoff)

    if seed_filter is not None:
        experiments = [e for e in experiments if e["seed"] in seed_filter]

    if not experiments:
        print("ERROR: No matching experiments found.")
        print("Expected: comparison_m10 dirs with BA topology, 200k timesteps")
        if args.ba_m is not None:
            print(f"Filtered to ba_m={args.ba_m}")
        return

    seeds = [e["seed"] for e in experiments]
    ba_m_values = sorted(set(e["run_config"].get("ba_m", "?") for e in experiments))
    print(f"Found {len(experiments)} experiments, seeds: {seeds}")
    print(f"ba_m values: {ba_m_values}")
    print()

    # Create output directory (ba_m/cutoff-specific if filtered)
    suffix = ""
    if args.ba_m is not None:
        suffix += f"_bam{args.ba_m}"
    if args.cutoff is not None:
        suffix += f"_auc{args.cutoff // 1000}k"
    output_dir = args.outputs_dir / f"seed_sweep_analysis{suffix}"
    output_dir.mkdir(exist_ok=True)

    # Tables
    print_per_seed_table(experiments, output_dir)
    print()
    print_grouped_summary(experiments, output_dir)
    print()
    print_graph_properties(experiments, output_dir)
    print()

    # Plots
    print("Generating plots...")
    plot_struct_gap_vs_gb_advantage(experiments, output_dir)
    plot_graph_property_vs_gb_advantage(experiments, output_dir)
    plot_avg_auc_by_method(experiments, output_dir)
    plot_gb_advantage_histogram(experiments, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
