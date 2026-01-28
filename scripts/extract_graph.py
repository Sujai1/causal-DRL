"""Extract and display causal graph from RDDL domain/instance."""

import argparse
import json
from pathlib import Path

from causal_fmdp_drl.graphs.extract_dbn import extract_causal_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = extract_causal_graph(args.domain, args.instance)

    print(f"State variables ({graph.num_vars}):")
    for v in graph.state_vars:
        parents = [graph.state_vars[i] for i in graph.parents(graph.state_vars.index(v))]
        print(f"  {v} <- {parents}")
    print(f"\nk_global: {graph.k_global}")
    print(f"density: {graph.density:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
