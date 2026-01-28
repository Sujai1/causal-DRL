"""Tests for CausalGraph dataclass."""

import numpy as np
import pytest

from causal_fmdp_drl.graphs.causal_graph import CausalGraph


def _make_chain_graph(n: int) -> CausalGraph:
    """Chain: 0->1->2->...->n-1."""
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return CausalGraph(state_vars=[f"x{i}" for i in range(n)], adjacency=adj)


def _make_star_graph(n: int) -> CausalGraph:
    """Star: node 0 is parent of all others."""
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        adj[0, i] = 1
    return CausalGraph(state_vars=[f"x{i}" for i in range(n)], adjacency=adj)


class TestCausalGraphProperties:
    def test_num_vars(self):
        g = _make_chain_graph(5)
        assert g.num_vars == 5

    def test_k_global_chain(self):
        g = _make_chain_graph(5)
        assert g.k_global == 1

    def test_k_global_star(self):
        g = _make_star_graph(5)
        assert g.k_global == 1  # each child has exactly 1 parent

    def test_k_global_dense(self):
        adj = np.ones((4, 4)) - np.eye(4)
        g = CausalGraph(state_vars=["a", "b", "c", "d"], adjacency=adj)
        assert g.k_global == 3  # each node has 3 parents

    def test_density_empty(self):
        adj = np.zeros((3, 3))
        g = CausalGraph(state_vars=["a", "b", "c"], adjacency=adj)
        assert g.density == 0.0

    def test_density_full(self):
        adj = np.ones((3, 3)) - np.eye(3)
        g = CausalGraph(state_vars=["a", "b", "c"], adjacency=adj)
        assert g.density == pytest.approx(1.0)

    def test_density_single_node(self):
        g = CausalGraph(state_vars=["a"], adjacency=np.zeros((1, 1)))
        assert g.density == 0.0

    def test_parents(self):
        g = _make_chain_graph(4)
        assert g.parents(0) == []
        assert g.parents(1) == [0]
        assert g.parents(2) == [1]
        assert g.parents(3) == [2]

    def test_K_causal(self):
        g = _make_chain_graph(5)
        assert g.K_causal(num_actions=10) == 10  # 10^1

        adj = np.ones((3, 3)) - np.eye(3)
        g2 = CausalGraph(state_vars=["a", "b", "c"], adjacency=adj)
        assert g2.K_causal(num_actions=2) == 4  # 2^2 (k_global=2)

    def test_to_dict(self):
        g = _make_chain_graph(3)
        d = g.to_dict()
        assert d["state_vars"] == ["x0", "x1", "x2"]
        assert d["k_global"] == 1
        assert isinstance(d["adjacency"], list)
        assert len(d["adjacency"]) == 3
