"""Generate RDDL instance files for SysAdmin with custom topologies."""

from pathlib import Path
from typing import Literal, Optional

import networkx as nx
import numpy as np


def generate_topology(
    num_machines: int,
    topology: Literal["ring", "star", "erdos_renyi", "barabasi_albert"],
    seed: int = 0,
    er_prob: float = 0.1,
    ba_m: int = 2,
) -> np.ndarray:
    """Generate adjacency matrix for machine connectivity.

    Returns:
        adj: (num_machines, num_machines) binary adjacency matrix
             (symmetric, no self-loops).
    """
    if topology == "ring":
        g = nx.cycle_graph(num_machines)
    elif topology == "star":
        g = nx.star_graph(num_machines - 1)
    elif topology == "erdos_renyi":
        g = nx.erdos_renyi_graph(num_machines, er_prob, seed=seed)
    elif topology == "barabasi_albert":
        g = nx.barabasi_albert_graph(num_machines, ba_m, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    adj = nx.to_numpy_array(g, dtype=np.float64)
    np.fill_diagonal(adj, 0)
    return adj


def write_sysadmin_instance(
    adj: np.ndarray,
    instance_name: str,
    output_dir: Path,
    horizon: int = 100,
    discount: float = 0.95,
    reboot_penalty: Optional[float] = None,
    reboot_prob: Optional[float] = None,
) -> Path:
    """Write a SysAdmin RDDL instance file.

    Uses 1-indexed computer names (c1, c2, ...) to match rddlrepository
    convention.

    Args:
        adj: Adjacency matrix for machine connectivity.
        instance_name: Name for the RDDL instance.
        output_dir: Directory to write the instance file.
        horizon: Max steps per episode.
        discount: Discount factor.
        reboot_penalty: Override REBOOT-PENALTY (domain default: 0.75).
        reboot_prob: Override REBOOT-PROB (domain default: 0.1).

    Returns:
        Path to the written instance file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize instance name for RDDL identifier validity
    import re
    instance_name = re.sub(r"[^a-zA-Z0-9_]", "_", instance_name)

    m = adj.shape[0]
    computers = [f"c{i + 1}" for i in range(m)]

    # Build CONNECTED non-fluents
    connected_lines = []
    for i in range(m):
        for j in range(m):
            if adj[i, j] == 1:
                connected_lines.append(
                    f"\t\tCONNECTED({computers[i]},{computers[j]});"
                )

    # Override domain defaults if specified
    if reboot_penalty is not None:
        connected_lines.append(f"\t\tREBOOT-PENALTY = {reboot_penalty};")
    if reboot_prob is not None:
        connected_lines.append(f"\t\tREBOOT-PROB = {reboot_prob};")

    # Build init-state (all running)
    init_lines = [f"\t\trunning({c});" for c in computers]

    instance_text = f"""non-fluents nf_{instance_name} {{
\tdomain = sysadmin_mdp;
\tobjects {{
\t\tcomputer : {{{",".join(computers)}}};
\t}};
\tnon-fluents {{
{chr(10).join(connected_lines)}
\t}};
}}

instance {instance_name} {{
\tdomain = sysadmin_mdp;
\tnon-fluents = nf_{instance_name};
\tinit-state {{
{chr(10).join(init_lines)}
\t}};

\tmax-nondef-actions = 1;
\thorizon  = {horizon};
\tdiscount = {discount};
}}
"""

    file_path = output_dir / f"{instance_name}.rddl"
    file_path.write_text(instance_text)
    return file_path
