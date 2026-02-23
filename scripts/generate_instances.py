"""Generate RDDL instance files for SysAdmin."""

import argparse
from pathlib import Path

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_machines", type=int, default=10)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument("--er_prob", type=float, default=0.2)
    parser.add_argument("--ba_m", type=int, default=2,
                        help="Number of edges to attach from new node (Barabási-Albert topology)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/rddl/sysadmin/instances"))
    args = parser.parse_args()

    adj = generate_topology(args.num_machines, args.topology, args.seed, args.er_prob, args.ba_m)
    name = f"{args.topology}_m{args.num_machines}_s{args.seed}"
    path = write_sysadmin_instance(adj, name, args.output_dir, args.horizon)

    print(f"Edges: {int(adj.sum())}, Shape: {adj.shape}")
    print(f"Instance written to {path}")


if __name__ == "__main__":
    main()
