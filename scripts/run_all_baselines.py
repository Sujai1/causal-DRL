"""Run all baselines on a single SysAdmin instance and save results.

Baselines:
  1. SB3 PPO
  2. SB3 DQN
  3. Custom DQN (no regularization)
  4. Custom DQN + causal rank regularization

All baselines share the same generated instance and causal graph.
Results are grouped under a single timestamped output directory.
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.graphs.extract_dbn import extract_causal_graph
from causal_fmdp_drl.agents.sb3_runner import train_sb3
from causal_fmdp_drl.agents.custom_dqn_runner import train_custom_dqn
from causal_fmdp_drl.agents.custom_dqn.agent import DQNConfig
from plot_results import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Run all baselines on a single SysAdmin instance."
    )
    parser.add_argument("--num_machines", type=int, default=10)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument("--er_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--lambda_reg", type=float, default=0.01,
                        help="Regularization strength for custom DQN + reg")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden layer width for all methods (2 layers)")
    args = parser.parse_args()

    # --- 1. Generate shared instance ---
    artifacts_dir = Path("artifacts/rddl/sysadmin")
    domain_path = artifacts_dir / "domain.rddl"
    adj = generate_topology(args.num_machines, args.topology, args.seed, args.er_prob)
    instance_path = write_sysadmin_instance(
        adj,
        f"{args.topology}_m{args.num_machines}_s{args.seed}",
        artifacts_dir / "instances",
        horizon=args.horizon,
    )

    # --- 2. Extract shared causal graph ---
    graph = extract_causal_graph(domain_path, instance_path)

    # --- 3. Create output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("outputs") / f"{timestamp}_comparison_m{args.num_machines}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save shared run config
    run_config = {
        **vars(args),
        "baselines": ["sb3_ppo", "sb3_dqn", "custom_dqn_noreg", "custom_dqn_reg"],
    }
    with open(base_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Save shared graph
    with open(base_dir / "graph.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    # Copy instance for reproducibility
    shutil.copy(instance_path, base_dir / "instance.rddl")

    # Print summary
    print(f"Topology: {args.topology}, machines: {args.num_machines}")
    print(f"k_global: {graph.k_global}")
    print(f"K_causal(m={args.num_machines}): {graph.K_causal(args.num_machines)}")
    print(f"Graph density: {graph.density:.3f}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Output: {base_dir}")
    print()

    # --- 4. Run each baseline ---
    baselines = [
        ("sb3_ppo", "SB3 PPO"),
        ("sb3_dqn", "SB3 DQN"),
        ("custom_dqn_noreg", "Custom DQN (no reg)"),
        ("custom_dqn_reg", f"Custom DQN (lambda={args.lambda_reg})"),
    ]

    wall_times = {}

    for key, label in baselines:
        output_dir = base_dir / key
        print(f"--- {label} ---")
        t0 = time.time()

        sb3_policy_kwargs = {
            "net_arch": [args.hidden_dim, args.hidden_dim],
        }

        if key == "sb3_ppo":
            train_sb3(
                algo="ppo",
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                policy_kwargs=sb3_policy_kwargs,
            )
        elif key == "sb3_dqn":
            train_sb3(
                algo="dqn",
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                policy_kwargs=sb3_policy_kwargs,
            )
        elif key == "custom_dqn_noreg":
            train_custom_dqn(
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                lambda_reg=0.0,
                dqn_config=DQNConfig(hidden_dim=args.hidden_dim),
            )
        elif key == "custom_dqn_reg":
            train_custom_dqn(
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                lambda_reg=args.lambda_reg,
                dqn_config=DQNConfig(hidden_dim=args.hidden_dim),
            )

        wall_times[key] = time.time() - t0
        print(f"  -> {output_dir} ({wall_times[key]:.1f}s)\n")

    # Save wall times to run_config
    run_config["wall_times"] = wall_times
    with open(base_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # --- 5. Summary ---
    print("=" * 60)
    print("All baselines complete.")
    print(f"Results: {base_dir}/")
    print()
    print("Directory structure:")
    print(f"  {base_dir}/")
    print(f"  ├── run_config.json")
    print(f"  ├── graph.json")
    print(f"  ├── instance.rddl")
    for key, _ in baselines:
        print(f"  ├── {key}/metrics.jsonl")
    print()

    # --- 6. Generate plots ---
    print("Generating plots...")
    generate_all_plots(base_dir)
    print()


if __name__ == "__main__":
    main()
