"""Run all baselines on a single SysAdmin instance and save results.

Baselines:
  1. Custom DQN (no reg, no LN) — vanilla reference
  2. Custom DQN + LN (no reg) — LayerNorm prevents rank collapse
  3. Custom DQN + GB + LN (k=k_causal, k_targets...) — gradient-balanced tail energy with LN
  4. SB3 PPO (optional)
  5. SB3 DQN (optional)
  6. Tabular Q-Learning (optional, if state space tractable)
  7. Dyna-Q (optional, if state space tractable)

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
from causal_fmdp_drl.graphs.causal_graph import CausalGraph
from causal_fmdp_drl.agents.sb3_runner import train_sb3
from causal_fmdp_drl.agents.custom_dqn_runner import train_custom_dqn
from causal_fmdp_drl.agents.custom_dqn.agent import DQNConfig
from causal_fmdp_drl.agents.tabular_runner import train_tabular_q, train_dyna_q
from causal_fmdp_drl.agents.tabular.state_encoding import check_tractable
from causal_fmdp_drl.agents.heuristic_runner import run_heuristic
from causal_fmdp_drl.agents.heuristic_policies import (
    noop_policy,
    random_reboot_policy,
    random_down_reboot_policy,
    highest_degree_down_policy,
    most_down_neighbors_policy,
    myopic_greedy_policy,
)
from plot_results import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Run all baselines on a single SysAdmin instance."
    )
    parser.add_argument("--num_machines", type=int, default=10)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument("--er_prob", type=float, default=0.2)
    parser.add_argument("--ba_m", type=int, default=2,
                        help="Number of edges to attach from new node (Barabási-Albert topology)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--lambda_reg", type=float, default=0.01,
                        help="Regularization strength for gradient-balanced methods")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden layer width for all methods (2 layers)")
    parser.add_argument("--planning_steps", type=int, default=10,
                        help="Planning steps per real step for Dyna-Q")
    parser.add_argument("--skip_tabular", action="store_true",
                        help="Skip tabular methods even if state space is tractable")
    parser.add_argument("--eps_decay_frac", type=float, default=0.5,
                        help="Fraction of timesteps over which to decay epsilon (tabular methods)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor for all methods (default: 0.95 to match RDDL instance)")
    parser.add_argument("--k_targets", type=int, nargs="+", default=None,
                        help="Additional k_target values for gradient-balanced. k_global is always "
                             "included automatically. Creates one baseline per unique k value.")
    parser.add_argument("--baselines", type=str, nargs="+", default=None,
                        choices=["sb3_ppo", "sb3_dqn",
                                 "custom_dqn_noreg", "custom_dqn_noreg_ln",
                                 "custom_dqn_gradient_balanced",
                                 "tabular_q", "dyna_q",
                                 "heuristic_noop", "heuristic_random_reboot",
                                 "heuristic_random_down",
                                 "heuristic_highest_degree",
                                 "heuristic_most_down_neighbors",
                                 "heuristic_myopic_greedy"],
                        help="Which baselines to run. If not specified, runs all. "
                             "custom_dqn_gradient_balanced expands based on --k_targets.")
    parser.add_argument("--gate_tau", type=float, default=0.005,
                        help="Soft gate threshold for gradient_balanced (default: 0.005 = 0.5%% tail energy). "
                             "Lower values make the gate turn off faster when constraint is satisfied.")
    parser.add_argument("--reg_warmup_frac", type=float, default=0.1,
                        help="Fraction of timesteps before regularization starts (default: 0.1 = 10%%). "
                             "During warmup, representation forms without regularization interference.")
    # Domain dynamics overrides
    parser.add_argument("--reboot_penalty", type=float, default=None,
                        help="Override REBOOT-PENALTY in domain (default: 0.75 from domain.rddl)")
    parser.add_argument("--reboot_prob", type=float, default=None,
                        help="Override REBOOT-PROB (spontaneous recovery rate) in domain (default: 0.1)")
    args = parser.parse_args()

    # --- 1. Generate shared instance ---
    artifacts_dir = Path("artifacts/rddl/sysadmin")
    domain_path = artifacts_dir / "domain.rddl"
    adj = generate_topology(args.num_machines, args.topology, args.seed, args.er_prob, args.ba_m)
    instance_path = write_sysadmin_instance(
        adj,
        f"{args.topology}_m{args.num_machines}_s{args.seed}",
        artifacts_dir / "instances",
        horizon=args.horizon,
        reboot_penalty=args.reboot_penalty,
        reboot_prob=args.reboot_prob,
    )

    # --- 2. Build shared causal graph from adjacency (no XADD needed) ---
    graph = CausalGraph.from_adjacency(adj, args.num_machines)

    # --- 3. Create output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("outputs") / f"{timestamp}_s{args.seed}_m{args.num_machines}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save shared run config (baselines list will be updated after running)
    run_config = {
        **vars(args),
        "baselines_ran": [],  # Will be populated after running
    }

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
    if args.baselines:
        print(f"Selected baselines: {', '.join(args.baselines)}")
    print(f"Output: {base_dir}")
    print()

    # --- 4. Check tabular tractability ---
    tabular_ok = check_tractable(args.num_machines) and not args.skip_tabular

    # --- 5. Determine which baselines to run ---
    selected = set(args.baselines) if args.baselines else None  # None means run all

    def should_run(key: str) -> bool:
        """Check if baseline should run based on --baselines flag and tractability."""
        if key in ("tabular_q", "dyna_q") and not tabular_ok:
            return False
        if selected is None:
            return True
        # Gradient-balanced variants expand from base name
        if key.startswith("custom_dqn_gradient_balanced"):
            return "custom_dqn_gradient_balanced" in selected
        return key in selected

    baselines = []

    # Core baselines
    if should_run("sb3_ppo"):
        baselines.append(("sb3_ppo", "SB3 PPO"))
    if should_run("sb3_dqn"):
        baselines.append(("sb3_dqn", "SB3 DQN"))

    # Custom DQN (no reg, no LN) — vanilla reference
    if should_run("custom_dqn_noreg"):
        baselines.append(("custom_dqn_noreg", "Custom DQN (no reg)"))

    # Custom DQN + LN (no reg)
    if should_run("custom_dqn_noreg_ln"):
        baselines.append(("custom_dqn_noreg_ln", "DQN + LN (no reg)"))

    # Gradient-balanced + LN baselines (one per k_target)
    if should_run("custom_dqn_gradient_balanced"):
        k_values = set(args.k_targets) if args.k_targets else {graph.k_global}
        for k in sorted(k_values):
            label_suffix = " (k_global)" if k == graph.k_global else ""
            baselines.append((
                f"custom_dqn_gradient_balanced_k{k}",
                f"GB+LN (k={k}{label_suffix})",
            ))

    # Tabular baselines
    if should_run("tabular_q"):
        baselines.append(("tabular_q", "Tabular Q-Learning"))
    if should_run("dyna_q"):
        baselines.append(("dyna_q", f"Dyna-Q (k={args.planning_steps})"))

    # Heuristic baselines
    if should_run("heuristic_noop"):
        baselines.append(("heuristic_noop", "No-Op"))
    if should_run("heuristic_random_reboot"):
        baselines.append(("heuristic_random_reboot", "Random Reboot (Any)"))
    if should_run("heuristic_random_down"):
        baselines.append(("heuristic_random_down", "Random Down Reboot"))
    if should_run("heuristic_highest_degree"):
        baselines.append(("heuristic_highest_degree", "Highest-Degree Down"))
    if should_run("heuristic_most_down_neighbors"):
        baselines.append(("heuristic_most_down_neighbors", "Most Down Neighbors"))
    if should_run("heuristic_myopic_greedy"):
        baselines.append(("heuristic_myopic_greedy", "Myopic Greedy"))

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
                graph=graph,
                policy_kwargs=sb3_policy_kwargs,
                gamma=args.gamma,
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
                graph=graph,
                policy_kwargs=sb3_policy_kwargs,
                gamma=args.gamma,
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
                gamma=args.gamma,
                use_layernorm=False,
                eps_decay_frac=args.eps_decay_frac,
                graph=graph,
            )
        elif key == "custom_dqn_noreg_ln":
            train_custom_dqn(
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                lambda_reg=0.0,
                dqn_config=DQNConfig(hidden_dim=args.hidden_dim),
                gamma=args.gamma,
                use_layernorm=True,
                eps_decay_frac=args.eps_decay_frac,
                graph=graph,
            )
        elif key.startswith("custom_dqn_gradient_balanced"):
            k_target = int(key.split("_k")[-1])
            reg_warmup = int(args.timesteps * args.reg_warmup_frac)
            train_custom_dqn(
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                lambda_reg=args.lambda_reg,
                reg_type="gradient_balanced",
                k_target=k_target,
                gate_tau=args.gate_tau,
                reg_warmup_steps=reg_warmup,
                dqn_config=DQNConfig(hidden_dim=args.hidden_dim),
                gamma=args.gamma,
                use_layernorm=True,
                eps_decay_frac=args.eps_decay_frac,
                graph=graph,
            )
        elif key == "tabular_q":
            train_tabular_q(
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                eps_decay_frac=args.eps_decay_frac,
                gamma=args.gamma,
                graph=graph,
            )
        elif key == "dyna_q":
            train_dyna_q(
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                planning_steps=args.planning_steps,
                eps_decay_frac=args.eps_decay_frac,
                gamma=args.gamma,
                graph=graph,
            )
        elif key.startswith("heuristic_"):
            import numpy as np
            heuristic_rng = np.random.default_rng(args.seed)
            reboot_prob_val = args.reboot_prob if args.reboot_prob is not None else 0.1
            reboot_penalty_val = args.reboot_penalty if args.reboot_penalty is not None else 0.75

            policy_map = {
                "heuristic_noop": (noop_policy, {
                    "num_machines": args.num_machines,
                }),
                "heuristic_random_reboot": (random_reboot_policy, {
                    "num_machines": args.num_machines,
                    "rng": heuristic_rng,
                }),
                "heuristic_random_down": (random_down_reboot_policy, {
                    "num_machines": args.num_machines,
                    "rng": heuristic_rng,
                }),
                "heuristic_highest_degree": (highest_degree_down_policy, {
                    "num_machines": args.num_machines,
                    "adj": adj,
                    "rng": heuristic_rng,
                }),
                "heuristic_most_down_neighbors": (most_down_neighbors_policy, {
                    "num_machines": args.num_machines,
                    "adj": adj,
                    "rng": heuristic_rng,
                }),
                "heuristic_myopic_greedy": (myopic_greedy_policy, {
                    "num_machines": args.num_machines,
                    "adj": adj,
                    "reboot_prob": reboot_prob_val,
                    "reboot_penalty": reboot_penalty_val,
                }),
            }
            policy_fn, policy_kwargs = policy_map[key]
            run_heuristic(
                policy_fn=policy_fn,
                policy_kwargs=policy_kwargs,
                domain_path=domain_path,
                instance_path=instance_path,
                output_dir=output_dir,
                total_timesteps=args.timesteps,
                max_episode_steps=args.horizon,
                seed=args.seed,
                graph=graph,
            )

        wall_times[key] = time.time() - t0
        print(f"  -> {output_dir} ({wall_times[key]:.1f}s)\n")

    # Save wall times and baselines to run_config
    run_config["baselines_ran"] = [k for k, _ in baselines]
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
