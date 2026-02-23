"""Train SB3 baseline agent."""

import argparse
from datetime import datetime
from pathlib import Path

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.agents.sb3_runner import train_sb3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    parser.add_argument("--num_machines", type=int, default=10)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument("--er_prob", type=float, default=0.2)
    parser.add_argument("--ba_m", type=int, default=2,
                        help="Number of edges to attach from new node (Barabási-Albert topology)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()

    artifacts_dir = Path("artifacts/rddl/sysadmin")
    adj = generate_topology(args.num_machines, args.topology, args.seed, args.er_prob, args.ba_m)
    instance_path = write_sysadmin_instance(
        adj,
        f"{args.topology}_m{args.num_machines}_s{args.seed}",
        artifacts_dir / "instances",
        horizon=args.horizon,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path("outputs")
        / f"{timestamp}_{args.algo}_{args.topology}_m{args.num_machines}"
    )

    train_sb3(
        algo=args.algo,
        domain_path=artifacts_dir / "domain.rddl",
        instance_path=instance_path,
        output_dir=output_dir,
        total_timesteps=args.timesteps,
        max_episode_steps=args.horizon,
        seed=args.seed,
    )

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
