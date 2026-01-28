"""Tests for DBN extraction from RDDL SysAdmin domain."""

from pathlib import Path

import numpy as np
import pytest

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.graphs.extract_dbn import extract_causal_graph

DOMAIN_PATH = Path("artifacts/rddl/sysadmin/domain.rddl")


@pytest.fixture
def ring_instance(tmp_path):
    adj = generate_topology(5, "ring")
    return write_sysadmin_instance(adj, "ring5", tmp_path, horizon=10)


@pytest.fixture
def star_instance(tmp_path):
    adj = generate_topology(5, "star")
    return write_sysadmin_instance(adj, "star5", tmp_path, horizon=10)


@pytest.mark.skipif(not DOMAIN_PATH.exists(), reason="domain.rddl not found")
class TestExtractCausalGraph:
    def test_returns_causal_graph(self, ring_instance):
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        assert graph.num_vars == 5

    def test_state_var_names(self, ring_instance):
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        assert all("running" in v for v in graph.state_vars)
        assert len(graph.state_vars) == 5

    def test_adjacency_shape(self, ring_instance):
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        assert graph.adjacency.shape == (5, 5)

    def test_ring_has_self_loops(self, ring_instance):
        """In SysAdmin, running(ci)' depends on running(ci) (self-dependency)."""
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        for i in range(5):
            assert graph.adjacency[i, i] == 1, f"Node {i} should have self-loop"

    def test_ring_has_neighbor_edges(self, ring_instance):
        """Ring: each node has edges from its ring neighbors."""
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        # In a ring of 5, each node's next-state depends on neighbors + self
        # k_global should be at least 2 (self + at least one neighbor)
        assert graph.k_global >= 2

    def test_star_center_has_many_parents(self, star_instance):
        """Star center should have edges from all leaf nodes."""
        graph = extract_causal_graph(DOMAIN_PATH, star_instance)
        # The center node (c1) is connected to all others
        # So running(c1)' depends on running(c1) + neighbors
        assert graph.k_global >= 2

    def test_k_global_positive(self, ring_instance):
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        assert graph.k_global > 0

    def test_density_reasonable(self, ring_instance):
        graph = extract_causal_graph(DOMAIN_PATH, ring_instance)
        assert 0.0 < graph.density <= 1.0
