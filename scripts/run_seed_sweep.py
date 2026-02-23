"""Run the same baseline comparison across multiple seeds."""

import argparse
import subprocess
import sys
import time


FIXED_ARGS = [
    "--num_machines", "10",
    "--topology", "barabasi_albert",
    "--ba_m", "2",
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
    "heuristic_myopic_greedy",
    "--k_targets", "8", "16",
    "--eps_decay_frac", "0.2",
    "--reboot_prob", "0.005",
    "--reboot_penalty", "1.75",
    "--hidden_dim", "128",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline comparison across multiple seeds."
    )
    parser.add_argument("--start_seed", type=int, default=13)
    parser.add_argument("--num_seeds", type=int, default=12)
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    total = len(seeds)
    sweep_start = time.time()

    print(f"Running {total} seeds: {seeds[0]}..{seeds[-1]}")
    print(f"Estimated time: ~{total * 10} minutes")
    print()

    for i, seed in enumerate(seeds):
        cmd = [
            sys.executable, "scripts/run_all_baselines.py",
            "--seed", str(seed),
            *FIXED_ARGS,
        ]
        print(f"{'='*60}")
        print(f"Seed {seed} ({i+1}/{total})")
        print(f"{'='*60}")

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"WARNING: seed {seed} exited with code {result.returncode}")

        print(f"Seed {seed} done in {elapsed/60:.1f} min")
        print()

    total_time = time.time() - sweep_start
    print(f"All {total} seeds complete in {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
