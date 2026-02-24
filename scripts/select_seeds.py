"""Screen seeds by graph hub structure and select top-N / bottom-N groups.

Generates BA graphs for many candidate seeds (instant — no RL),
computes hub metrics, and selects two maximally separated groups:
  - high-hub: top N seeds by hub_edge_frac
  - low-hub: bottom N seeds by hub_edge_frac

Tracks the highest seed ever used in outputs/seed_tracker.json to
prevent accidental reuse across experiments.

Usage:
    python scripts/select_seeds.py                    # screen 100 seeds, pick top/bottom 10
    python scripts/select_seeds.py --n 5 --pool 200   # screen 200 seeds, pick top/bottom 5
    python scripts/select_seeds.py --dry_run           # show candidates without updating tracker
"""

import argparse
import json
from pathlib import Path

import numpy as np

from causal_fmdp_drl.envs.rddl.instance_generator import generate_topology


TRACKER_PATH = Path("outputs/seed_tracker.json")


# ---------------------------------------------------------------------------
# Seed tracker
# ---------------------------------------------------------------------------

def load_tracker() -> dict:
    """Load seed tracker, initializing from existing experiments if needed."""
    if TRACKER_PATH.exists():
        return json.loads(TRACKER_PATH.read_text())
    # Bootstrap from existing experiment directories
    return _bootstrap_tracker()


def _bootstrap_tracker() -> dict:
    """Scan existing experiments to find all seeds ever used."""
    outputs_dir = Path("outputs")
    used_seeds = set()
    for exp_dir in outputs_dir.iterdir():
        rc = exp_dir / "run_config.json"
        if rc.exists():
            try:
                cfg = json.loads(rc.read_text())
                seed = cfg.get("seed")
                if seed is not None:
                    used_seeds.add(seed)
            except (json.JSONDecodeError, KeyError):
                continue

    highest = max(used_seeds) if used_seeds else 0
    tracker = {
        "highest_seed_used": highest,
        "used_seeds": sorted(used_seeds),
    }
    # Save the bootstrapped tracker
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2))
    print(f"Bootstrapped seed tracker from existing experiments: "
          f"{len(used_seeds)} seeds found, highest={highest}")
    return tracker


def save_tracker(tracker: dict) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2))


def register_seeds(seeds: list[int]) -> None:
    """Add seeds to the tracker as used."""
    tracker = load_tracker()
    used = set(tracker.get("used_seeds", []))
    used.update(seeds)
    tracker["used_seeds"] = sorted(used)
    tracker["highest_seed_used"] = max(tracker.get("highest_seed_used", 0), max(seeds))
    save_tracker(tracker)


# ---------------------------------------------------------------------------
# Graph screening
# ---------------------------------------------------------------------------

def compute_hub_metrics(adj: np.ndarray) -> dict:
    """Compute hub-related metrics from adjacency matrix."""
    n = adj.shape[0]
    deg = adj.sum(axis=1) - np.diag(adj)
    hub_mask = deg >= 5
    total_edges = adj.sum() - np.trace(adj)
    hub_edges = adj[hub_mask].sum() - np.diag(adj)[hub_mask].sum()

    sorted_deg = np.sort(deg)[::-1]
    deg_sum = deg.sum()

    return {
        "num_hubs": int(hub_mask.sum()),
        "hub_edge_frac": float(hub_edges / total_edges) if total_edges > 0 else 0.0,
        "top2_deg_share": float(sorted_deg[:2].sum() / deg_sum) if deg_sum > 0 else 0.0,
        "deg_max": float(deg.max()),
        "deg_std": float(deg.std()),
    }


def screen_seeds(
    start_seed: int,
    pool_size: int,
    num_machines: int = 10,
    ba_m: int = 2,
) -> list[dict]:
    """Generate BA graphs for pool_size consecutive seeds, return metrics."""
    candidates = []
    for seed in range(start_seed, start_seed + pool_size):
        adj = generate_topology(num_machines, "barabasi_albert", seed=seed, ba_m=ba_m)
        metrics = compute_hub_metrics(adj)
        metrics["seed"] = seed
        candidates.append(metrics)
    return candidates


def select_top_bottom(
    candidates: list[dict],
    n: int,
    sort_key: str = "hub_edge_frac",
) -> tuple[list[dict], list[dict]]:
    """Select top-N and bottom-N candidates by sort_key."""
    ranked = sorted(candidates, key=lambda c: c[sort_key], reverse=True)
    top = ranked[:n]
    bottom = ranked[-n:]
    return top, bottom


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_selection(top: list[dict], bottom: list[dict]) -> None:
    header = (
        f"{'Seed':>5}  {'hub_e_frac':>10}  {'num_hubs':>8}  "
        f"{'top2_share':>10}  {'deg_max':>7}  {'deg_std':>7}"
    )
    sep = "-" * len(header)

    print("HIGH-HUB GROUP")
    print(header)
    print(sep)
    for c in sorted(top, key=lambda x: x["hub_edge_frac"], reverse=True):
        print(
            f"{c['seed']:>5}  {c['hub_edge_frac']:>10.3f}  {c['num_hubs']:>8}  "
            f"{c['top2_deg_share']:>10.3f}  {c['deg_max']:>7.0f}  {c['deg_std']:>7.2f}"
        )
    top_seeds = sorted(c["seed"] for c in top)
    print(f"Seeds: {top_seeds}")
    print()

    print("LOW-HUB GROUP")
    print(header)
    print(sep)
    for c in sorted(bottom, key=lambda x: x["hub_edge_frac"]):
        print(
            f"{c['seed']:>5}  {c['hub_edge_frac']:>10.3f}  {c['num_hubs']:>8}  "
            f"{c['top2_deg_share']:>10.3f}  {c['deg_max']:>7.0f}  {c['deg_std']:>7.2f}"
        )
    bottom_seeds = sorted(c["seed"] for c in bottom)
    print(f"Seeds: {bottom_seeds}")
    print()

    # Summary stats
    for label, group in [("High-hub", top), ("Low-hub", bottom)]:
        hef = [c["hub_edge_frac"] for c in group]
        nh = [c["num_hubs"] for c in group]
        t2 = [c["top2_deg_share"] for c in group]
        print(
            f"{label}: hub_edge_frac={np.mean(hef):.3f}±{np.std(hef):.3f}  "
            f"num_hubs={np.mean(nh):.1f}±{np.std(nh):.1f}  "
            f"top2_share={np.mean(t2):.3f}±{np.std(t2):.3f}"
        )


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

BASELINE_ARGS = [
    "--topology", "barabasi_albert",
    "--timesteps", "200000",
    "--horizon", "100",
    "--gamma", "0.99",
    "--baselines",
    "custom_dqn_noreg_ln",
    "custom_dqn_gradient_balanced",
    "heuristic_noop",
    "heuristic_random_reboot",
    "heuristic_random_down",
    "heuristic_highest_degree",
    "heuristic_most_down_neighbors",
    "--k_targets", "8", "16",
    "--eps_decay_frac", "0.2",
    "--reboot_prob", "0.005",
    "--reboot_penalty", "1.75",
    "--hidden_dim", "128",
]


def run_group(
    seeds: list[int],
    label: str,
    num_machines: int,
    ba_m: int,
) -> None:
    """Run experiments for a list of seeds sequentially."""
    import subprocess
    import sys
    import time

    total = len(seeds)
    print(f"\n{'='*60}")
    print(f"Running {label} ({total} seeds): {seeds}")
    print(f"{'='*60}\n")

    group_start = time.time()
    for i, seed in enumerate(seeds):
        cmd = [
            sys.executable, "scripts/run_all_baselines.py",
            "--seed", str(seed),
            "--ba_m", str(ba_m),
            "--num_machines", str(num_machines),
            *BASELINE_ARGS,
        ]
        print(f"--- {label}: seed {seed} ({i+1}/{total}) ---")
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"WARNING: seed {seed} exited with code {result.returncode}")
        print(f"Seed {seed} done in {elapsed/60:.1f} min\n")

    total_time = time.time() - group_start
    print(f"{label} complete: {total} seeds in {total_time/60:.1f} min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Screen seeds by hub structure, select top/bottom groups."
    )
    parser.add_argument("--n", type=int, default=10,
                        help="Number of seeds per group (default: 10)")
    parser.add_argument("--pool", type=int, default=100,
                        help="Number of candidate seeds to screen (default: 100)")
    parser.add_argument("--ba_m", type=int, default=2,
                        help="BA attachment parameter (default: 2)")
    parser.add_argument("--num_machines", type=int, default=10)
    parser.add_argument("--run", action="store_true",
                        help="Run experiments on selected seeds (default: just show selection)")
    parser.add_argument("--group", choices=["both", "high", "low"], default="both",
                        help="Which group to run: both, high, or low (default: both)")
    args = parser.parse_args()

    # Load tracker to find starting seed
    tracker = load_tracker()
    start_seed = tracker["highest_seed_used"] + 1
    print(f"Highest seed previously used: {tracker['highest_seed_used']}")
    print(f"Screening seeds {start_seed}..{start_seed + args.pool - 1} "
          f"(pool={args.pool}, ba_m={args.ba_m}, m={args.num_machines})")
    print()

    # Screen
    candidates = screen_seeds(start_seed, args.pool, args.num_machines, args.ba_m)

    # Select
    top, bottom = select_top_bottom(candidates, args.n)
    print_selection(top, bottom)

    # Separation check
    top_min = min(c["hub_edge_frac"] for c in top)
    bottom_max = max(c["hub_edge_frac"] for c in bottom)
    print(f"\nSeparation: top min={top_min:.3f}, bottom max={bottom_max:.3f}, "
          f"gap={top_min - bottom_max:.3f}")

    top_seeds = sorted(c["seed"] for c in top)
    bottom_seeds = sorted(c["seed"] for c in bottom)
    all_selected = sorted(top_seeds + bottom_seeds)
    print(f"\nAll selected seeds: {all_selected}")

    # Register seeds
    register_seeds(all_selected)
    updated = load_tracker()
    print(f"Seed tracker updated. New highest: {updated['highest_seed_used']}")

    if not args.run:
        print("\nTo run experiments, add --run")
        print(f"  python scripts/select_seeds.py --n {args.n} --pool {args.pool} --run")
        print(f"  python scripts/select_seeds.py --n {args.n} --pool {args.pool} --run --group high")
        print(f"  python scripts/select_seeds.py --n {args.n} --pool {args.pool} --run --group low")
        return

    # Run experiments
    if args.group in ("both", "high"):
        run_group(top_seeds, "HIGH-HUB", args.num_machines, args.ba_m)
    if args.group in ("both", "low"):
        run_group(bottom_seeds, "LOW-HUB", args.num_machines, args.ba_m)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
