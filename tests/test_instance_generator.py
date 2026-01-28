"""Tests for topology generation and RDDL instance writing."""

import numpy as np
import pytest

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)


class TestGenerateTopology:
    def test_ring_shape(self):
        adj = generate_topology(10, "ring")
        assert adj.shape == (10, 10)

    def test_ring_symmetric(self):
        adj = generate_topology(10, "ring")
        np.testing.assert_array_equal(adj, adj.T)

    def test_ring_no_self_loops(self):
        adj = generate_topology(10, "ring")
        assert np.diag(adj).sum() == 0

    def test_ring_edge_count(self):
        adj = generate_topology(10, "ring")
        assert adj.sum() == 20  # 10 edges, each counted twice (symmetric)

    def test_star_shape(self):
        adj = generate_topology(5, "star")
        assert adj.shape == (5, 5)

    def test_star_center_degree(self):
        adj = generate_topology(5, "star")
        assert adj[0].sum() == 4  # center connected to all others

    def test_erdos_renyi_shape(self):
        adj = generate_topology(10, "erdos_renyi", seed=42, er_prob=0.2)
        assert adj.shape == (10, 10)

    def test_erdos_renyi_symmetric(self):
        adj = generate_topology(10, "erdos_renyi", seed=42, er_prob=0.2)
        np.testing.assert_array_equal(adj, adj.T)

    def test_erdos_renyi_deterministic(self):
        a1 = generate_topology(10, "erdos_renyi", seed=42, er_prob=0.2)
        a2 = generate_topology(10, "erdos_renyi", seed=42, er_prob=0.2)
        np.testing.assert_array_equal(a1, a2)

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            generate_topology(5, "unknown")


class TestWriteSysadminInstance:
    def test_file_created(self, tmp_path):
        adj = generate_topology(5, "ring")
        path = write_sysadmin_instance(adj, "test_inst", tmp_path)
        assert path.exists()

    def test_file_contains_domain(self, tmp_path):
        adj = generate_topology(5, "ring")
        path = write_sysadmin_instance(adj, "test_inst", tmp_path)
        text = path.read_text()
        assert "sysadmin_mdp" in text

    def test_file_contains_computers(self, tmp_path):
        adj = generate_topology(5, "ring")
        path = write_sysadmin_instance(adj, "test_inst", tmp_path)
        text = path.read_text()
        for i in range(1, 6):
            assert f"c{i}" in text

    def test_file_contains_connected(self, tmp_path):
        adj = np.zeros((3, 3))
        adj[0, 1] = 1
        adj[1, 0] = 1
        path = write_sysadmin_instance(adj, "test_inst", tmp_path)
        text = path.read_text()
        assert "CONNECTED(c1,c2)" in text
        assert "CONNECTED(c2,c1)" in text

    def test_file_contains_horizon(self, tmp_path):
        adj = generate_topology(3, "ring")
        path = write_sysadmin_instance(adj, "test_inst", tmp_path, horizon=50)
        text = path.read_text()
        assert "horizon  = 50" in text

    def test_file_contains_max_nondef_actions(self, tmp_path):
        adj = generate_topology(3, "ring")
        path = write_sysadmin_instance(adj, "test_inst", tmp_path)
        text = path.read_text()
        assert "max-nondef-actions = 1" in text
