"""Tests for CausalGraph.from_adjacency and equivalence with extract_causal_graph."""

import numpy as np
import pytest

from causal_fmdp_drl.graphs.causal_graph import CausalGraph
from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)


class TestFromAdjacency:
    """Unit tests for CausalGraph.from_adjacency."""

    def test_variable_names(self):
        adj = np.zeros((3, 3))
        graph = CausalGraph.from_adjacency(adj, 3)
        assert graph.state_vars == ["running___c1", "running___c2", "running___c3"]

    def test_self_loops_added(self):
        adj = np.zeros((3, 3))
        graph = CausalGraph.from_adjacency(adj, 3)
        # With no network edges, each node should only have a self-loop
        assert np.array_equal(graph.adjacency, np.eye(3))
        assert graph.k_global == 1

    def test_network_edges_preserved(self):
        # Ring: 0-1, 1-2, 2-0
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.float64)
        graph = CausalGraph.from_adjacency(adj, 3)
        # Causal adjacency should be all ones (network edges + self-loops)
        expected = np.ones((3, 3), dtype=np.float64)
        assert np.array_equal(graph.adjacency, expected)
        assert graph.k_global == 3

    def test_star_topology(self):
        # Star with 4 nodes: node 0 connected to 1, 2, 3
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = 1
        adj[0, 2] = adj[2, 0] = 1
        adj[0, 3] = adj[3, 0] = 1
        graph = CausalGraph.from_adjacency(adj, 4)
        # Node 0: self + 3 neighbors = 4 parents
        assert graph.k_global == 4
        # Leaf nodes: self + hub = 2 parents
        assert len(graph.parents(1)) == 2
        assert len(graph.parents(2)) == 2
        assert len(graph.parents(3)) == 2

    def test_num_vars(self):
        adj = np.zeros((5, 5))
        graph = CausalGraph.from_adjacency(adj, 5)
        assert graph.num_vars == 5

    def test_clipping_idempotent(self):
        # If adj already has self-loops, result should be same as without
        adj_no_self = np.array([[0, 1], [1, 0]], dtype=np.float64)
        adj_with_self = np.array([[1, 1], [1, 1]], dtype=np.float64)
        g1 = CausalGraph.from_adjacency(adj_no_self, 2)
        g2 = CausalGraph.from_adjacency(adj_with_self, 2)
        assert np.array_equal(g1.adjacency, g2.adjacency)


class TestFromAdjacencyMatchesExtraction:
    """Compare from_adjacency output against extract_causal_graph (XADD).

    These tests require pyRDDLGym_symbolic and are slow (~30s each).
    Mark with pytest.mark.slow so they can be skipped in CI.
    """

    @pytest.fixture
    def er_instance(self, tmp_path):
        """Generate an Erdos-Renyi instance and return (adj, domain_path, instance_path)."""
        num_machines = 5
        seed = 42
        adj = generate_topology(num_machines, "erdos_renyi", seed, er_prob=0.4)
        artifacts_dir = tmp_path / "rddl" / "sysadmin"
        # Copy domain file to tmp
        import shutil
        from pathlib import Path
        src_domain = Path("artifacts/rddl/sysadmin/domain.rddl")
        dest_domain = artifacts_dir / "domain.rddl"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_domain, dest_domain)

        instance_path = write_sysadmin_instance(
            adj,
            f"test_er_m{num_machines}_s{seed}",
            artifacts_dir / "instances",
        )
        return adj, num_machines, dest_domain, instance_path

    @pytest.mark.slow
    def test_adjacency_matches(self, er_instance):
        """from_adjacency adjacency must match extract_causal_graph adjacency."""
        from causal_fmdp_drl.graphs.extract_dbn import extract_causal_graph

        adj, num_machines, domain_path, instance_path = er_instance
        graph_xadd = extract_causal_graph(domain_path, instance_path)
        graph_fast = CausalGraph.from_adjacency(adj, num_machines)

        assert graph_fast.state_vars == graph_xadd.state_vars
        np.testing.assert_array_equal(graph_fast.adjacency, graph_xadd.adjacency)

    @pytest.mark.slow
    def test_k_global_matches(self, er_instance):
        """k_global must match between the two methods."""
        from causal_fmdp_drl.graphs.extract_dbn import extract_causal_graph

        adj, num_machines, domain_path, instance_path = er_instance
        graph_xadd = extract_causal_graph(domain_path, instance_path)
        graph_fast = CausalGraph.from_adjacency(adj, num_machines)

        assert graph_fast.k_global == graph_xadd.k_global
