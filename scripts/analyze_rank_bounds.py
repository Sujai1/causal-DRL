"""Analyze rank-bound (gradient-balanced) experiments across seeds.

Questions:
1. Which k values perform best vs DQN+LN across seeds?
2. Do graph characteristics predict which k is best?
3. Can we relate this to causal structure / Bellman rank?
"""

import json
import re
import sys
from pathlib import Path

import numpy as np


def load_seed_experiment(exp_dir: Path) -> dict | None:
    """Load summary, graph, and config for one seed experiment."""
    summary_path = exp_dir / "summary.json"
    graph_path = exp_dir / "graph.json"
    config_path = exp_dir / "run_config.json"

    if not all(p.exists() for p in [summary_path, graph_path, config_path]):
        return None

    with open(summary_path) as f:
        summary = json.load(f)
    with open(graph_path) as f:
        graph = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    return {"summary": summary, "graph": graph, "config": config, "dir": str(exp_dir)}


def compute_graph_characteristics(graph: dict) -> dict:
    """Compute graph characteristics from adjacency matrix."""
    adj = np.array(graph["adjacency"])
    n = adj.shape[0]
    k_global = graph.get("k_global", None)
    density = graph.get("density", None)

    # Degree statistics (row sums minus self-loops)
    degrees = adj.sum(axis=1) - np.diag(adj)  # out-degree (neighbors)
    in_degrees = adj.sum(axis=0) - np.diag(adj)

    max_degree = float(degrees.max())
    min_degree = float(degrees.min())
    mean_degree = float(degrees.mean())
    std_degree = float(degrees.std())
    degree_skew = float(((degrees - mean_degree) ** 3).mean() / (std_degree ** 3 + 1e-8))

    # Hub concentration: fraction of edges touching the max-degree node
    hub_edge_frac = float(max_degree / degrees.sum()) if degrees.sum() > 0 else 0.0

    # Number of "hub" nodes (degree > mean + 1 std)
    num_hubs = int((degrees > mean_degree + std_degree).sum())

    # Spectral properties of adjacency
    eigenvalues = np.linalg.eigvalsh(adj)
    spectral_gap = float(eigenvalues[-1] - eigenvalues[-2]) if n > 1 else 0.0
    spectral_radius = float(abs(eigenvalues).max())

    # Effective rank of adjacency matrix (via entropy of normalized singular values)
    svs = np.linalg.svd(adj, compute_uv=False)
    svs_norm = svs / (svs.sum() + 1e-12)
    adj_effective_rank = float(np.exp(-np.sum(svs_norm * np.log(svs_norm + 1e-12))))

    # Clustering coefficient (average)
    clustering_coeffs = []
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        neighbors = neighbors[neighbors != i]
        k_i = len(neighbors)
        if k_i < 2:
            clustering_coeffs.append(0.0)
            continue
        # Count edges among neighbors
        subgraph = adj[np.ix_(neighbors, neighbors)]
        actual_edges = (subgraph.sum() - np.trace(subgraph)) / 2
        possible_edges = k_i * (k_i - 1) / 2
        clustering_coeffs.append(float(actual_edges / possible_edges))
    avg_clustering = float(np.mean(clustering_coeffs))

    return {
        "k_global": k_global,
        "density": density,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "mean_degree": mean_degree,
        "std_degree": std_degree,
        "degree_skew": degree_skew,
        "hub_edge_frac": hub_edge_frac,
        "num_hubs": num_hubs,
        "spectral_gap": spectral_gap,
        "spectral_radius": spectral_radius,
        "adj_effective_rank": adj_effective_rank,
        "avg_clustering": avg_clustering,
    }


def extract_seed_from_dir(exp_dir: Path) -> int | None:
    """Extract seed number from directory name like ..._s143_m10."""
    match = re.search(r"_s(\d+)_m", exp_dir.name)
    return int(match.group(1)) if match else None


def is_compatible_config(config: dict) -> bool:
    """Check if experiment config matches current standard."""
    return (
        config.get("topology") == "barabasi_albert"
        and config.get("timesteps") == 200000
        and config.get("gamma") == 0.99
        and config.get("hidden_dim") == 128
        and config.get("num_machines") == 10
    )


def main():
    outputs_dir = Path("outputs")

    # Find ALL experiment directories (seed-based and comparison-based)
    exp_dirs = sorted(outputs_dir.glob("*_m10"))
    if not exp_dirs:
        print("No experiments found in outputs/")
        sys.exit(1)

    print(f"Scanning {len(exp_dirs)} experiment directories...\n")

    # Collect data — deduplicate by seed (prefer newer experiments)
    records_by_seed = {}
    for exp_dir in exp_dirs:
        data = load_seed_experiment(exp_dir)
        if data is None:
            continue

        config = data["config"]
        if not is_compatible_config(config):
            continue

        seed = config.get("seed")
        if seed is None:
            seed = extract_seed_from_dir(exp_dir)
        if seed is None:
            continue

        graph_chars = compute_graph_characteristics(data["graph"])

        summary = data["summary"]

        # Extract DQN+LN baseline
        ln_data = summary.get("custom_dqn_noreg_ln")
        if ln_data is None:
            continue

        # Extract all gradient_balanced variants
        gb_variants = {}
        for key, val in summary.items():
            match = re.match(r"custom_dqn_gradient_balanced_k(\d+)", key)
            if match:
                k = int(match.group(1))
                gb_variants[k] = val

        if not gb_variants:
            continue

        record = {
            "seed": seed,
            "graph_chars": graph_chars,
            "ln": ln_data,
            "gb_variants": gb_variants,
            "dir": str(exp_dir),
        }

        # Keep the record with most GB variants (prefer richer experiments)
        if seed not in records_by_seed or len(gb_variants) > len(records_by_seed[seed]["gb_variants"]):
            records_by_seed[seed] = record

    records = list(records_by_seed.values())

    if not records:
        print("No valid experiment data found")
        sys.exit(1)

    print(f"Loaded {len(records)} unique seed experiments with gradient-balanced variants")

    # ========================================================
    # 1. Overall performance comparison: each k vs DQN+LN
    # ========================================================
    print("=" * 80)
    print("1. GRADIENT-BALANCED (k) vs DQN+LN: FINAL MEAN RETURN")
    print("=" * 80)

    k_values = sorted(set(k for r in records for k in r["gb_variants"]))

    print(f"\n{'Seed':>6}  {'k_global':>8}  {'DQN+LN':>8}", end="")
    for k in k_values:
        print(f"  {'k='+str(k):>8}", end="")
    print(f"  {'Best k':>8}  {'Best Δ':>8}")
    print("-" * (40 + 10 * len(k_values) + 20))

    wins_by_k = {k: 0 for k in k_values}
    delta_by_k = {k: [] for k in k_values}
    best_k_list = []

    for r in sorted(records, key=lambda x: x["seed"]):
        ln_ret = r["ln"]["final_mean_return"]
        best_k = None
        best_ret = ln_ret
        best_delta = 0.0

        print(f"{r['seed']:>6}  {r['graph_chars']['k_global']:>8}  {ln_ret:>8.1f}", end="")
        for k in k_values:
            if k in r["gb_variants"]:
                gb_ret = r["gb_variants"][k]["final_mean_return"]
                delta = gb_ret - ln_ret
                delta_by_k[k].append(delta)
                print(f"  {gb_ret:>8.1f}", end="")
                if gb_ret > best_ret:
                    best_ret = gb_ret
                    best_k = k
                    best_delta = delta
            else:
                print(f"  {'N/A':>8}", end="")

        if best_k is not None:
            wins_by_k[best_k] += 1
            best_k_list.append((r["seed"], best_k, best_delta, r["graph_chars"]))
            print(f"  {'k='+str(best_k):>8}  {best_delta:>+8.1f}")
        else:
            best_k_list.append((r["seed"], None, 0.0, r["graph_chars"]))
            print(f"  {'LN wins':>8}  {0.0:>+8.1f}")

    # Summary stats
    print(f"\n{'':>6}  {'':>8}  {'':>8}", end="")
    for k in k_values:
        deltas = delta_by_k[k]
        if deltas:
            mean_d = np.mean(deltas)
            print(f"  {mean_d:>+8.1f}", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print("  (mean Δ vs LN)")

    print(f"\nWin counts (seeds where k=X has highest final_mean_return):")
    ln_wins = sum(1 for _, bk, _, _ in best_k_list if bk is None)
    print(f"  DQN+LN: {ln_wins}")
    for k in k_values:
        print(f"  k={k}: {wins_by_k[k]}")

    # ========================================================
    # 2. AUC comparison (cumulative learning performance)
    # ========================================================
    print("\n" + "=" * 80)
    print("2. GRADIENT-BALANCED (k) vs DQN+LN: AUC (CUMULATIVE LEARNING)")
    print("=" * 80)

    auc_delta_by_k = {k: [] for k in k_values}
    auc_wins_by_k = {k: 0 for k in k_values}

    print(f"\n{'Seed':>6}  {'DQN+LN':>10}", end="")
    for k in k_values:
        print(f"  {'k='+str(k):>10}", end="")
    print(f"  {'Best k':>8}")
    print("-" * (20 + 12 * len(k_values) + 10))

    for r in sorted(records, key=lambda x: x["seed"]):
        ln_auc = r["ln"]["auc"]
        best_k = None
        best_auc = ln_auc

        print(f"{r['seed']:>6}  {ln_auc:>10.0f}", end="")
        for k in k_values:
            if k in r["gb_variants"]:
                gb_auc = r["gb_variants"][k]["auc"]
                auc_delta_by_k[k].append(gb_auc - ln_auc)
                print(f"  {gb_auc:>10.0f}", end="")
                if gb_auc > best_auc:
                    best_auc = gb_auc
                    best_k = k
            else:
                print(f"  {'N/A':>10}", end="")

        if best_k is not None:
            auc_wins_by_k[best_k] += 1
            print(f"  {'k='+str(best_k):>8}")
        else:
            print(f"  {'LN wins':>8}")

    print(f"\nAUC win counts:")
    auc_ln_wins = len(records) - sum(auc_wins_by_k.values())
    print(f"  DQN+LN: {auc_ln_wins}")
    for k in k_values:
        mean_auc_d = np.mean(auc_delta_by_k[k]) if auc_delta_by_k[k] else 0
        print(f"  k={k}: {auc_wins_by_k[k]} wins, mean AUC Δ = {mean_auc_d:+.0f}")

    # ========================================================
    # 3. Graph characteristics vs best k
    # ========================================================
    print("\n" + "=" * 80)
    print("3. GRAPH CHARACTERISTICS vs BEST PERFORMING k")
    print("=" * 80)

    char_names = [
        "k_global", "density", "max_degree", "mean_degree", "std_degree",
        "degree_skew", "hub_edge_frac", "num_hubs", "spectral_gap",
        "spectral_radius", "adj_effective_rank", "avg_clustering",
    ]

    print(f"\n{'Seed':>6}  {'Best k':>8}  {'Δ ret':>8}", end="")
    for cn in ["k_global", "max_deg", "hub_frac", "spect_gap", "eff_rank", "clust"]:
        print(f"  {cn:>9}", end="")
    print()
    print("-" * (30 + 11 * 6))

    for seed, best_k, best_delta, gc in sorted(best_k_list):
        k_str = f"k={best_k}" if best_k else "LN"
        print(
            f"{seed:>6}  {k_str:>8}  {best_delta:>+8.1f}"
            f"  {gc['k_global']:>9}"
            f"  {gc['max_degree']:>9.0f}"
            f"  {gc['hub_edge_frac']:>9.3f}"
            f"  {gc['spectral_gap']:>9.2f}"
            f"  {gc['adj_effective_rank']:>9.2f}"
            f"  {gc['avg_clustering']:>9.3f}"
        )

    # ========================================================
    # 4. Correlation analysis
    # ========================================================
    print("\n" + "=" * 80)
    print("4. CORRELATIONS: Graph characteristic vs advantage of best GB over LN")
    print("=" * 80)

    # For each seed, compute max advantage of any GB over LN
    advantages = []
    char_values = {cn: [] for cn in char_names}
    best_k_values_for_corr = []

    for r in records:
        ln_ret = r["ln"]["final_mean_return"]
        max_advantage = 0.0
        best_k_ret = None
        for k in k_values:
            if k in r["gb_variants"]:
                delta = r["gb_variants"][k]["final_mean_return"] - ln_ret
                if delta > max_advantage:
                    max_advantage = delta
                    best_k_ret = k
        advantages.append(max_advantage)
        best_k_values_for_corr.append(best_k_ret if best_k_ret else 0)
        for cn in char_names:
            char_values[cn].append(r["graph_chars"][cn])

    advantages = np.array(advantages)

    print(f"\n{'Characteristic':>20}  {'Corr w/ advantage':>18}  {'Corr w/ best k':>15}")
    print("-" * 58)
    for cn in char_names:
        vals = np.array(char_values[cn], dtype=float)
        if np.std(vals) < 1e-10:
            print(f"{cn:>20}  {'(constant)':>18}  {'(constant)':>15}")
            continue
        corr_adv = np.corrcoef(vals, advantages)[0, 1]
        corr_k = np.corrcoef(vals, best_k_values_for_corr)[0, 1]
        print(f"{cn:>20}  {corr_adv:>+18.3f}  {corr_k:>+15.3f}")

    # ========================================================
    # 5. k_global / k_target ratio analysis
    # ========================================================
    print("\n" + "=" * 80)
    print("5. k_target / k_global RATIO ANALYSIS")
    print("=" * 80)

    print(f"\nDoes the ratio k_target/k_global predict performance?")
    print(f"(k_global is the causal graph's max in-degree + 1)\n")

    for r in sorted(records, key=lambda x: x["seed"]):
        k_g = r["graph_chars"]["k_global"]
        ln_ret = r["ln"]["final_mean_return"]
        print(f"Seed {r['seed']} (k_global={k_g}):")

        # Sort GB variants by return
        variants = []
        for k, v in r["gb_variants"].items():
            ratio = k / k_g if k_g > 0 else float('inf')
            delta = v["final_mean_return"] - ln_ret
            variants.append((k, ratio, v["final_mean_return"], delta))
        variants.sort(key=lambda x: -x[2])  # sort by return descending

        for k, ratio, ret, delta in variants:
            marker = " *** BEST" if delta == max(v[3] for v in variants) and delta > 0 else ""
            print(f"  k={k:>3} (ratio={ratio:>5.1f}x): return={ret:>6.1f}  Δ={delta:>+6.1f}{marker}")
        print()

    # ========================================================
    # 6. Summary statistics
    # ========================================================
    print("=" * 80)
    print("6. SUMMARY")
    print("=" * 80)

    any_gb_beats_ln = sum(1 for a in advantages if a > 0)
    print(f"\nSeeds where ANY gradient-balanced k beats DQN+LN: {any_gb_beats_ln}/{len(records)}")
    print(f"Mean best advantage: {advantages.mean():+.2f}")
    print(f"Max advantage: {advantages.max():+.2f}")

    # Best k across seeds by mean final return
    print(f"\nMean final return across all seeds:")
    ln_returns = [r["ln"]["final_mean_return"] for r in records]
    print(f"  DQN+LN: {np.mean(ln_returns):.2f} ± {np.std(ln_returns):.2f}")
    for k in k_values:
        k_returns = [r["gb_variants"][k]["final_mean_return"] for r in records if k in r["gb_variants"]]
        if k_returns:
            print(f"  k={k:>3}: {np.mean(k_returns):.2f} ± {np.std(k_returns):.2f}  (Δ={np.mean(k_returns) - np.mean(ln_returns):+.2f})")

    # k_global distribution
    k_globals = [r["graph_chars"]["k_global"] for r in records]
    print(f"\nk_global distribution: {sorted(k_globals)}")
    print(f"  mean={np.mean(k_globals):.1f}, min={min(k_globals)}, max={max(k_globals)}")


if __name__ == "__main__":
    main()
