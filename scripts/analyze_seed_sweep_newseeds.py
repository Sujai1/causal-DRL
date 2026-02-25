"""Analyze seed sweep results with dynamic k_target detection.

Supports all k_targets (k=8, 16, 32, 48, 64, 80, 96, 112, etc.)
and outputs to *_newseeds directories to avoid overwriting old analyses.

Usage:
    python scripts/analyze_seed_sweep_newseeds.py --seeds 143-150
    python scripts/analyze_seed_sweep_newseeds.py --seeds 143-150 --cutoff 100000
"""

import argparse
import json
import re
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
# Data collection (reused from original, with _m10 fix)
# ---------------------------------------------------------------------------

def _matches_config(run_config: dict, ba_m: int | None = None) -> bool:
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
    adj = np.array(graph_data["adjacency"])
    n = adj.shape[0]
    degrees = adj.sum(axis=1) - np.diag(adj)
    sorted_deg = np.sort(degrees)[::-1]
    deg_sum = degrees.sum()

    gini = _gini_coefficient(degrees)
    hub_mask = degrees >= 5
    hub_edges = adj[hub_mask].sum() - np.diag(adj)[hub_mask].sum()
    total_edges = adj.sum() - np.trace(adj)
    hub_edge_frac = float(hub_edges / total_edges) if total_edges > 0 else 0.0
    top2_share = float(sorted_deg[:2].sum() / deg_sum) if deg_sum > 0 else 0.0
    top3_share = float(sorted_deg[:3].sum() / deg_sum) if deg_sum > 0 else 0.0
    hub_ratio = float(degrees.max() / degrees.mean()) if degrees.mean() > 0 else 0.0
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
        "deg_range": float(degrees.max() - degrees.min()),
        "deg_entropy": deg_entropy,
        "num_hubs": int(np.sum(hub_mask)),
        "num_low_deg": int(np.sum(degrees <= 2)),
        "hub_ratio": hub_ratio,
        "hub_edge_frac": hub_edge_frac,
        "top2_deg_share": top2_share,
        "top3_deg_share": top3_share,
        "density": float(total_edges) / (n * (n - 1)),
        "k_global": graph_data.get("k_global"),
    }


def _gini_coefficient(values: np.ndarray) -> float:
    vals = np.sort(values)
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * vals) - (n + 1) * vals.sum()) / (n * vals.sum()))


def _load_metrics_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _auc_at_cutoff(records: list[dict], cutoff: int) -> float | None:
    filtered = [
        (r["timestep"], r["episode_return"])
        for r in records if r["timestep"] <= cutoff
    ]
    if len(filtered) < 2:
        return None
    ts, rets = zip(*filtered)
    return float(np.trapezoid(rets, ts))


def _recompute_summary_at_cutoff(exp_dir: Path, cutoff: int) -> dict:
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
            filtered_returns = [
                r["episode_return"] for r in records if r["timestep"] <= cutoff
            ]
            final_mean = (
                float(np.mean(filtered_returns[-10:]))
                if len(filtered_returns) >= 10
                else float(np.mean(filtered_returns))
            )
            summary[sub.name] = {
                "auc": auc,
                "final_mean_return": final_mean,
                "num_episodes": len(filtered_returns),
            }
    return summary


def collect_experiments(
    outputs_dir: Path, ba_m: int | None = None, cutoff: int | None = None,
) -> list[dict]:
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
        if not _matches_config(run_config, ba_m=ba_m):
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
        seen_seeds[seed] = entry

    return sorted(seen_seeds.values(), key=lambda e: e["seed"])


# ---------------------------------------------------------------------------
# Dynamic k_target detection
# ---------------------------------------------------------------------------

def detect_k_targets(experiments: list[dict]) -> list[int]:
    """Find all k_target values present across experiments."""
    k_set = set()
    for exp in experiments:
        for key in exp["summary"]:
            m = re.match(r"custom_dqn_gradient_balanced_k(\d+)", key)
            if m:
                k_set.add(int(m.group(1)))
    return sorted(k_set)


def _gb_key(k: int) -> str:
    return f"custom_dqn_gradient_balanced_k{k}"


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def _get_auc(summary: dict, baseline: str) -> float | None:
    entry = summary.get(baseline)
    return entry.get("auc") if entry else None


def _get_final_mean(summary: dict, baseline: str) -> float | None:
    entry = summary.get(baseline)
    return entry.get("final_mean_return") if entry else None


def _gb_vs_ln_pct(summary: dict, gb_key: str) -> float | None:
    ln_auc = _get_auc(summary, "custom_dqn_noreg_ln")
    gb_auc = _get_auc(summary, gb_key)
    if ln_auc is None or gb_auc is None or abs(ln_auc) < 1e-9:
        return None
    return (gb_auc - ln_auc) / abs(ln_auc) * 100


def _struct_aware_gap_pct(summary: dict) -> float | None:
    random_down_auc = _get_auc(summary, "heuristic_random_down")
    highest_deg_auc = _get_auc(summary, "heuristic_highest_degree")
    most_down_auc = _get_auc(summary, "heuristic_most_down_neighbors")
    if random_down_auc is None:
        return None
    candidates = [v for v in [highest_deg_auc, most_down_auc] if v is not None]
    if not candidates:
        return None
    struct_aware_auc = max(candidates)
    if abs(random_down_auc) < 1e-9:
        return None
    return (struct_aware_auc - random_down_auc) / abs(random_down_auc) * 100


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_per_seed_table(
    experiments: list[dict], k_targets: list[int], output_dir: Path,
) -> None:
    lines = []
    lines.append("=" * 140)
    lines.append("PER-SEED BREAKDOWN: GB vs DQN+LN (AUC gap %)")
    lines.append("=" * 140)
    lines.append("")

    # Header
    hdr = f"{'Seed':>4}  {'k_gl':>4}  {'deg_std':>7}  {'struct%':>7}"
    for k in k_targets:
        hdr += f"  {'k'+str(k)+'%':>8}"
    hdr += f"  {'best_k':>6}  {'best%':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for exp in experiments:
        s = exp["summary"]
        gs = exp["graph_stats"]
        struct_gap = _struct_aware_gap_pct(s)

        row = f"{exp['seed']:>4}  {gs.get('k_global', '?'):>4}  {gs['deg_std']:>7.2f}"
        row += f"  {struct_gap:>7.1f}%" if struct_gap is not None else f"  {'N/A':>7}"

        best_k = None
        best_pct = None
        for k in k_targets:
            pct = _gb_vs_ln_pct(s, _gb_key(k))
            if pct is not None:
                row += f"  {pct:>+7.1f}%"
                if best_pct is None or pct > best_pct:
                    best_pct = pct
                    best_k = k
            else:
                row += f"  {'N/A':>8}"

        if best_pct is not None and best_pct > 0:
            row += f"  {'k'+str(best_k):>6}  {best_pct:>+6.1f}%"
        else:
            row += f"  {'LN':>6}  {'0.0':>7}"
        lines.append(row)

    # Summary row: mean gap per k
    mean_row = f"{'MEAN':>4}  {'':>4}  {'':>7}  {'':>7}"
    for k in k_targets:
        pcts = [
            _gb_vs_ln_pct(e["summary"], _gb_key(k))
            for e in experiments
        ]
        valid = [p for p in pcts if p is not None]
        if valid:
            mean_row += f"  {np.mean(valid):>+7.1f}%"
        else:
            mean_row += f"  {'N/A':>8}"
    lines.append("")
    lines.append(mean_row + "  (mean)")

    # Win count row
    win_row = f"{'WINS':>4}  {'':>4}  {'':>7}  {'':>7}"
    for k in k_targets:
        wins = sum(
            1 for e in experiments
            if (_gb_vs_ln_pct(e["summary"], _gb_key(k)) or -1) > 0
        )
        win_row += f"  {f'{wins}/{len(experiments)}':>8}"
    lines.append(win_row + "  (GB wins)")

    lines.append("")
    content = "\n".join(lines)
    print(content)
    (output_dir / "per_seed_table.txt").write_text(content)


def print_summary_stats(
    experiments: list[dict], k_targets: list[int], output_dir: Path,
) -> None:
    lines = []
    lines.append("=" * 100)
    lines.append("SUMMARY STATISTICS: Mean AUC and Final Return across seeds")
    lines.append("=" * 100)
    lines.append("")

    # AUC table
    lines.append(f"{'Method':<35}  {'Mean AUC':>12}  {'Std AUC':>10}  {'n':>3}")
    lines.append("-" * 65)

    auc_by_method = {}
    for exp in experiments:
        for baseline, data in exp["summary"].items():
            auc = data.get("auc")
            if auc is not None:
                auc_by_method.setdefault(baseline, []).append(auc)

    sorted_methods = sorted(
        auc_by_method.keys(),
        key=lambda b: np.mean(auc_by_method[b]),
        reverse=True,
    )
    for method in sorted_methods:
        vals = auc_by_method[method]
        lines.append(
            f"{_label(method):<35}  {np.mean(vals):>12.0f}  {np.std(vals):>10.0f}  {len(vals):>3}"
        )

    # Final return table
    lines.append("")
    lines.append(f"{'Method':<35}  {'Mean Return':>12}  {'Std Return':>10}  {'n':>3}")
    lines.append("-" * 65)

    ret_by_method = {}
    for exp in experiments:
        for baseline, data in exp["summary"].items():
            ret = data.get("final_mean_return")
            if ret is not None:
                ret_by_method.setdefault(baseline, []).append(ret)

    sorted_ret = sorted(
        ret_by_method.keys(),
        key=lambda b: np.mean(ret_by_method[b]),
        reverse=True,
    )
    for method in sorted_ret:
        vals = ret_by_method[method]
        lines.append(
            f"{_label(method):<35}  {np.mean(vals):>12.1f}  {np.std(vals):>10.1f}  {len(vals):>3}"
        )

    lines.append("")
    content = "\n".join(lines)
    print(content)
    (output_dir / "summary_stats.txt").write_text(content)


def print_graph_properties(
    experiments: list[dict], k_targets: list[int], output_dir: Path,
) -> None:
    lines = []
    lines.append("=" * 120)
    lines.append("GRAPH PROPERTIES BY SEED (with best-k winner)")
    lines.append("=" * 120)
    lines.append("")

    hdr = (
        f"{'Seed':>4}  {'best_k':>6}  {'k_gl':>4}  {'deg_min':>7}  {'deg_max':>7}  "
        f"{'deg_mean':>8}  {'deg_std':>7}  {'#hubs≥5':>7}  {'#low≤2':>6}  "
        f"{'density':>7}  {'Gini':>5}  {'hub_frac':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    # Per-k win/loss buckets for graph stats
    gb_stats_by_k = {k: {"GB": [], "LN": []} for k in k_targets}

    for exp in experiments:
        gs = exp["graph_stats"]
        s = exp["summary"]

        # Find best k
        best_k = None
        best_pct = 0.0
        for k in k_targets:
            pct = _gb_vs_ln_pct(s, _gb_key(k))
            if pct is not None and pct > best_pct:
                best_pct = pct
                best_k = k

        best_str = f"k={best_k}" if best_k else "LN"
        lines.append(
            f"{exp['seed']:>4}  {best_str:>6}  {gs.get('k_global', '?'):>4}  "
            f"{gs['deg_min']:>7.1f}  {gs['deg_max']:>7.1f}  "
            f"{gs['deg_mean']:>8.2f}  {gs['deg_std']:>7.2f}  "
            f"{gs['num_hubs']:>7}  {gs['num_low_deg']:>6}  "
            f"{gs['density']:>7.3f}  {gs['deg_gini']:>5.3f}  "
            f"{gs['hub_edge_frac']:>8.3f}"
        )

        for k in k_targets:
            pct = _gb_vs_ln_pct(s, _gb_key(k))
            winner = "GB" if pct is not None and pct > 0 else "LN"
            gb_stats_by_k[k][winner].append(gs)

    # Grouped means per k
    lines.append("")
    lines.append("-" * len(hdr))
    for k in k_targets:
        for winner_label in ["GB", "LN"]:
            bucket = gb_stats_by_k[k][winner_label]
            n = len(bucket)
            if n == 0:
                lines.append(f"k={k} {winner_label} wins (n=0): no data")
                continue
            lines.append(
                f"k={k} {winner_label} wins (n={n}):  "
                f"deg_mean={np.mean([g['deg_mean'] for g in bucket]):.2f}  "
                f"deg_std={np.mean([g['deg_std'] for g in bucket]):.2f}  "
                f"#hubs={np.mean([g['num_hubs'] for g in bucket]):.1f}  "
                f"Gini={np.mean([g['deg_gini'] for g in bucket]):.3f}"
            )

    lines.append("")
    content = "\n".join(lines)
    print(content)
    (output_dir / "graph_properties.txt").write_text(content)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Distinct colors for up to 9 k values
_K_COLORS = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#e41a1c",
]
_K_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "*"]


def plot_avg_auc_by_method(
    experiments: list[dict], output_dir: Path,
) -> None:
    auc_by_baseline: dict[str, list[float]] = {}
    for exp in experiments:
        for baseline, data in exp["summary"].items():
            auc = data.get("auc")
            if auc is not None:
                auc_by_baseline.setdefault(baseline, []).append(auc)

    if not auc_by_baseline:
        return

    sorted_baselines = sorted(
        auc_by_baseline.keys(),
        key=lambda b: np.mean(auc_by_baseline[b]),
        reverse=True,
    )

    means = [np.mean(auc_by_baseline[b]) for b in sorted_baselines]
    stds = [np.std(auc_by_baseline[b]) for b in sorted_baselines]
    labels = [_label(b) for b in sorted_baselines]
    colors = [_color(b) for b in sorted_baselines]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sorted_baselines))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean AUC (across seeds)")
    ax.set_title(f"Average AUC by Method (n={len(experiments)} seeds)")
    ax.grid(True, alpha=0.3, axis="y")

    for i, b in enumerate(sorted_baselines):
        n = len(auc_by_baseline[b])
        if n < len(experiments):
            ax.annotate(f"n={n}", (i, means[i] + stds[i]),
                        ha="center", va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(output_dir / "avg_auc_by_method.png", dpi=150)
    plt.close(fig)
    print(f"  Saved avg_auc_by_method.png")


def plot_gb_advantage_histogram(
    experiments: list[dict], k_targets: list[int], output_dir: Path,
) -> None:
    n_k = len(k_targets)
    ncols = min(4, n_k)
    nrows = (n_k + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    if n_k == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, k in enumerate(k_targets):
        ax = axes_flat[idx]
        color = _K_COLORS[idx % len(_K_COLORS)]
        gaps = [
            g for g in (_gb_vs_ln_pct(e["summary"], _gb_key(k)) for e in experiments)
            if g is not None
        ]
        if not gaps:
            ax.set_title(f"k={k} — no data")
            continue

        gaps_arr = np.array(gaps)
        n_wins = int(np.sum(gaps_arr > 0))
        n_total = len(gaps_arr)

        ax.hist(gaps_arr, bins=max(6, n_total // 2), color=color,
                edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(gaps_arr), color="blue", linestyle="--", linewidth=1,
                   label=f"Mean={np.mean(gaps_arr):+.1f}%")
        ax.annotate(
            f"GB wins: {n_wins}/{n_total}",
            xy=(0.95, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7),
        )
        ax.set_xlabel(f"AUC Gap % over LN")
        ax.set_title(f"k={k}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n_k, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Grad-Bal Advantage Distribution by k_target", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "gb_advantage_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved gb_advantage_distribution.png")


def plot_struct_gap_vs_gb_advantage(
    experiments: list[dict], k_targets: list[int], output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    all_x, all_y = [], []

    for idx, k in enumerate(k_targets):
        color = _K_COLORS[idx % len(_K_COLORS)]
        marker = _K_MARKERS[idx % len(_K_MARKERS)]
        x_vals, y_vals, seed_labels = [], [], []
        for exp in experiments:
            sg = _struct_aware_gap_pct(exp["summary"])
            gap = _gb_vs_ln_pct(exp["summary"], _gb_key(k))
            if sg is not None and gap is not None:
                x_vals.append(sg)
                y_vals.append(gap)
                seed_labels.append(str(exp["seed"]))

        if not x_vals:
            continue

        x, y = np.array(x_vals), np.array(y_vals)
        all_x.extend(x_vals)
        all_y.extend(y_vals)

        ax.scatter(x, y, s=50, zorder=5, color=color, edgecolors="white",
                   linewidth=0.4, marker=marker, label=f"k={k}")

        for xi, yi, lbl in zip(x, y, seed_labels):
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color="#555555")

        if len(x) >= 3:
            rho, pval = spearmanr(x, y)
            star = "*" if pval < 0.05 else ""
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(min(all_x) - 1, max(all_x) + 1, 50)
            ax.plot(x_line, np.polyval(coeffs, x_line), "--", color=color, alpha=0.4)

    ax.axhline(0, color="red", linestyle=":", alpha=0.5, label="Break-even")
    ax.set_xlabel("Structure-Aware Heuristic Gap % (over Random Down)")
    ax.set_ylabel("Grad-Bal AUC Gap % (over DQN+LN)")
    ax.set_title("Grad-Bal Advantage vs Structure-Aware Heuristic Gap")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "struct_gap_vs_gb_advantage.png", dpi=150)
    plt.close(fig)
    print(f"  Saved struct_gap_vs_gb_advantage.png")


def plot_graph_property_vs_gb_advantage(
    experiments: list[dict], k_targets: list[int], output_dir: Path,
) -> None:
    x_axis_specs = [
        ("num_hubs", "# Hubs (deg >= 5)", False),
        ("hub_ratio", "Hub Ratio", False),
        ("hub_edge_frac", "Hub Edge Fraction", False),
        ("top2_deg_share", "Top-2 Deg Share", False),
        ("deg_std", "Degree Std Dev", False),
        ("deg_range", "Degree Range", False),
        ("deg_gini", "Gini Coefficient", False),
        ("deg_entropy", "Degree Entropy", False),
        ("deg_max", "Max Degree", False),
        ("num_low_deg", "# Low-Degree (≤2)", False),
        ("struct_gap", "Struct-Aware Gap %", True),
    ]

    n_panels = len(x_axis_specs)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = axes.flatten()

    for panel_idx, (key, xlabel, is_derived) in enumerate(x_axis_specs):
        ax = axes_flat[panel_idx]

        for kidx, k in enumerate(k_targets):
            color = _K_COLORS[kidx % len(_K_COLORS)]
            marker = _K_MARKERS[kidx % len(_K_MARKERS)]
            x_vals, y_vals = [], []
            for exp in experiments:
                gap = _gb_vs_ln_pct(exp["summary"], _gb_key(k))
                if gap is None:
                    continue
                xval = (
                    _struct_aware_gap_pct(exp["summary"])
                    if is_derived
                    else exp["graph_stats"].get(key)
                )
                if xval is None:
                    continue
                x_vals.append(xval)
                y_vals.append(gap)

            if len(x_vals) < 3:
                continue

            x, y = np.array(x_vals), np.array(y_vals)
            ax.scatter(x, y, s=30, zorder=5, color=color, edgecolors="white",
                       linewidth=0.3, marker=marker, label=f"k={k}")

            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, np.polyval(coeffs, x_line), "--", color=color, alpha=0.4)

            rho, pval = spearmanr(x, y)
            star = "*" if pval < 0.05 else ""
            y_pos = 0.95 - kidx * 0.08
            ax.annotate(
                f"k={k}: ρ={rho:.2f}{star}",
                xy=(0.03, y_pos), xycoords="axes fraction",
                verticalalignment="top", fontsize=7, color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="wheat", alpha=0.4),
            )

        ax.axhline(0, color="red", linestyle=":", alpha=0.4)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("GB AUC Gap % over LN", fontsize=9)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=6, ncol=2)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Grad-Bal Advantage vs Graph Properties (all k_targets)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "graph_properties_vs_gb_advantage.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved graph_properties_vs_gb_advantage.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze seed sweep with dynamic k_target detection."
    )
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ba_m", type=int, default=None)
    parser.add_argument("--cutoff", type=int, default=None,
                        help="Recompute AUC up to this timestep (e.g. 100000)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Seed range (e.g. 143-150) or comma-separated list")
    args = parser.parse_args()

    # Parse --seeds: supports "143-150", "1,2,3", or "143-150,243-258"
    seed_filter = None
    if args.seeds is not None:
        seed_filter = set()
        for part in args.seeds.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                seed_filter.update(range(int(lo), int(hi) + 1))
            else:
                seed_filter.add(int(part))

    cutoff_label = f", AUC@{args.cutoff // 1000}k" if args.cutoff else ""
    seed_label = f", seeds={args.seeds}" if args.seeds else ""
    print(f"Collecting experiments{cutoff_label}{seed_label}...")
    experiments = collect_experiments(args.outputs_dir, ba_m=args.ba_m, cutoff=args.cutoff)

    if seed_filter is not None:
        experiments = [e for e in experiments if e["seed"] in seed_filter]

    if not experiments:
        print("ERROR: No matching experiments found.")
        return

    k_targets = detect_k_targets(experiments)
    seeds = [e["seed"] for e in experiments]
    print(f"Found {len(experiments)} experiments, seeds: {seeds}")
    print(f"Detected k_targets: {k_targets}")
    print()

    # Output dir with _newseeds suffix
    suffix = "_newseeds"
    if args.ba_m is not None:
        suffix += f"_bam{args.ba_m}"
    if args.cutoff is not None:
        suffix += f"_auc{args.cutoff // 1000}k"
    output_dir = args.outputs_dir / f"seed_sweep_analysis{suffix}"
    output_dir.mkdir(exist_ok=True)

    # Tables
    print_per_seed_table(experiments, k_targets, output_dir)
    print()
    print_summary_stats(experiments, k_targets, output_dir)
    print()
    print_graph_properties(experiments, k_targets, output_dir)
    print()

    # Plots
    print("Generating plots...")
    plot_avg_auc_by_method(experiments, output_dir)
    plot_gb_advantage_histogram(experiments, k_targets, output_dir)
    plot_struct_gap_vs_gb_advantage(experiments, k_targets, output_dir)
    plot_graph_property_vs_gb_advantage(experiments, k_targets, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
