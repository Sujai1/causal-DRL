"""Extract causal graph from RDDL using pyRDDLGym-symbolic."""

from pathlib import Path

import numpy as np
from pyRDDLGym_symbolic.core.visualizer import RDDL2Graph

from .causal_graph import CausalGraph


def extract_causal_graph(
    domain_path: Path,
    instance_path: Path,
    state_fluent_filter: str = "running",
) -> CausalGraph:
    """Extract DBN structure and return CausalGraph for state fluents.

    The adjacency matrix includes edges from both state variables and
    action variables to next-state variables. This means k_global
    reflects the total in-degree (state + action parents) which is
    the quantity needed for the rank bound K_causal = |A|^{k_global}.

    Args:
        domain_path: Path to RDDL domain file.
        instance_path: Path to RDDL instance file.
        state_fluent_filter: Only keep state fluents containing this string.

    Returns:
        CausalGraph with state variable names and adjacency matrix.
    """
    r2g = RDDL2Graph(
        domain=domain_path.stem,
        domain_file=str(domain_path),
        instance_file=str(instance_path),
        directed=True,
    )

    model = r2g.model

    # Collect next-state CPFs for state fluents matching the filter.
    # Next-state variables end with "'" in the cpfs dict.
    next_state_cpfs = {}
    for gvar, node_id in model.cpfs.items():
        if gvar == "reward":
            continue
        # gvar looks like "running___c1'" — strip the prime to get state var name
        if not gvar.endswith("'"):
            continue
        base_var = gvar[:-1]  # e.g. "running___c1"
        if state_fluent_filter not in base_var:
            continue
        next_state_cpfs[base_var] = node_id

    state_vars = sorted(next_state_cpfs.keys())
    var_to_idx = {v: i for i, v in enumerate(state_vars)}
    n = len(state_vars)
    adjacency = np.zeros((n, n), dtype=np.float64)

    for var, node_id in next_state_cpfs.items():
        parents = model.collect_vars(node_id)
        j = var_to_idx[var]
        for parent in parents:
            # Only include state variable parents that match our filter
            if parent in var_to_idx:
                i = var_to_idx[parent]
                adjacency[i, j] = 1

    return CausalGraph(state_vars=state_vars, adjacency=adjacency)
