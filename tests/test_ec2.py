"""Tests for scripts/ec2.py helper functions."""

import sys
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ec2 import _parse_seed_range, _build_parallel_script, DEFAULT_BASELINE_FLAGS

SAMPLE_FLAGS = " ".join(DEFAULT_BASELINE_FLAGS)


class TestParseSeedRange:
    def test_simple_range(self):
        assert _parse_seed_range("3-7") == [3, 4, 5, 6, 7]

    def test_single_seed(self):
        assert _parse_seed_range("42") == [42]

    def test_comma_separated(self):
        assert _parse_seed_range("3,5,7") == [3, 5, 7]

    def test_mixed_ranges_and_singles(self):
        assert _parse_seed_range("3-5,10,15-17") == [3, 4, 5, 10, 15, 16, 17]

    def test_deduplicates_and_sorts(self):
        assert _parse_seed_range("5,3,5,1-3") == [1, 2, 3, 5]

    def test_single_element_range(self):
        assert _parse_seed_range("10-10") == [10]


class TestBuildParallelScript:
    def test_contains_all_seeds(self):
        script = _build_parallel_script([3, 5, 7], 2, 2, 10, SAMPLE_FLAGS)
        assert "seeds=(3 5 7)" in script

    def test_parallel_count(self):
        script = _build_parallel_script([1, 2, 3, 4], 4, 2, 10, SAMPLE_FLAGS)
        assert "parallel=4" in script

    def test_uses_correct_python(self):
        script = _build_parallel_script([1], 1, 2, 10, SAMPLE_FLAGS)
        assert "python3.10" in script

    def test_uses_ba_m(self):
        script = _build_parallel_script([1], 1, 3, 10, SAMPLE_FLAGS)
        assert "--ba_m 3" in script

    def test_uses_num_machines(self):
        script = _build_parallel_script([1], 1, 2, 20, SAMPLE_FLAGS)
        assert "--num_machines 20" in script

    def test_runs_run_all_baselines(self):
        script = _build_parallel_script([1], 1, 2, 10, SAMPLE_FLAGS)
        assert "run_all_baselines.py" in script

    def test_logs_per_seed(self):
        script = _build_parallel_script([10, 20], 1, 2, 10, SAMPLE_FLAGS)
        assert "seed_${seed}.log" in script

    def test_waits_for_batch(self):
        script = _build_parallel_script([1, 2, 3, 4], 2, 2, 10, SAMPLE_FLAGS)
        assert "wait $pid" in script

    def test_custom_flags(self):
        custom = "--topology star --timesteps 100"
        script = _build_parallel_script([1], 1, 2, 10, custom)
        assert "--topology star --timesteps 100" in script
        assert "heuristic_myopic_greedy" not in script

    def test_default_flags_include_myopic_greedy(self):
        script = _build_parallel_script([1], 1, 2, 10, SAMPLE_FLAGS)
        assert "heuristic_myopic_greedy" in script

    def test_default_flags_k_targets(self):
        script = _build_parallel_script([1], 1, 2, 10, SAMPLE_FLAGS)
        assert "--k_targets 16 32 48 64 80 96 112" in script
