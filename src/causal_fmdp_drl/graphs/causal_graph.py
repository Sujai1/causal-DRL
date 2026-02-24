"""Causal graph representation extracted from a Dynamic Bayesian Network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CausalGraph:
    """Ground-truth causal structure from DBN.

    Attributes:
        state_vars: Ordered list of state variable names.
        adjacency: (n, n) binary matrix where adj[i, j] = 1 means
            variable i is a parent of variable j in the DBN.
    """

    state_vars: List[str]
    adjacency: np.ndarray

    @classmethod
    def from_adjacency(cls, network_adj: np.ndarray, num_machines: int) -> CausalGraph:
        """Build CausalGraph from the network adjacency matrix.

        The causal (DBN) adjacency adds self-loops to the network topology:
        each machine's next-state depends on its own current state plus
        its network neighbours' states.

        Args:
            network_adj: (num_machines, num_machines) binary adjacency matrix
                of the network topology (no self-loops expected).
            num_machines: Number of machines (used for variable naming).

        Returns:
            CausalGraph with state_vars and causal adjacency.
        """
        state_vars = [f"running___c{i+1}" for i in range(num_machines)]
        causal_adj = np.clip(network_adj + np.eye(num_machines), 0, 1).astype(np.float64)
        return cls(state_vars=state_vars, adjacency=causal_adj)

    @property
    def num_vars(self) -> int:
        return len(self.state_vars)

    @property
    def k_global(self) -> int:
        """Maximum in-degree across all state variables."""
        return int(self.adjacency.sum(axis=0).max())

    @property
    def density(self) -> float:
        """Edge density: num_edges / max_possible_edges."""
        n = self.num_vars
        if n <= 1:
            return 0.0
        return float(self.adjacency.sum()) / (n * (n - 1))

    def parents(self, var_idx: int) -> List[int]:
        """Return indices of parents for variable at var_idx."""
        return np.where(self.adjacency[:, var_idx])[0].tolist()

    def K_causal(self, num_actions: int) -> int:
        """Compute rank bound: num_actions ** k_global."""
        return num_actions ** self.k_global

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "state_vars": self.state_vars,
            "adjacency": self.adjacency.tolist(),
            "k_global": self.k_global,
            "density": self.density,
        }
