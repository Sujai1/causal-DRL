"""Train custom DQN with optional causal rank regularization."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.envs.make_env import make_sysadmin_env
from causal_fmdp_drl.agents.custom_dqn_runner import train_custom_dqn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_machines", type=int, default=10)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument("--er_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()

    # Generate instance
    artifacts_dir = Path("artifacts/rddl/sysadmin")
    adj = generate_topology(args.num_machines, args.topology, args.seed, args.er_prob)
    instance_path = write_sysadmin_instance(
        adj,
        f"{args.topology}_m{args.num_machines}_s{args.seed}",
        artifacts_dir / "instances",
        horizon=args.horizon,
    )

    # Print causal graph info
    _, graph = make_sysadmin_env(
        artifacts_dir / "domain.rddl", instance_path,
        max_episode_steps=args.horizon, seed=args.seed,
    )
    print(f"k_global: {graph.k_global}")
    print(f"K_causal(m={args.num_machines}): {graph.K_causal(args.num_machines)}")
    print(f"Graph density: {graph.density:.3f}")
    print(f"lambda_reg: {args.lambda_reg}")

    # Output dir
    reg_str = f"reg{args.lambda_reg}" if args.lambda_reg > 0 else "noreg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path("outputs")
        / f"{timestamp}_dqn_{args.topology}_m{args.num_machines}_{reg_str}"
    )

    train_custom_dqn(
        domain_path=artifacts_dir / "domain.rddl",
        instance_path=instance_path,
        output_dir=output_dir,
        total_timesteps=args.timesteps,
        max_episode_steps=args.horizon,
        seed=args.seed,
        lambda_reg=args.lambda_reg,
    )

    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
