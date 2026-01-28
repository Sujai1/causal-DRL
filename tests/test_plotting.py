"""Smoke tests for plot_results.py — verify plots generate without errors."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path so we can import plot_results
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from plot_results import (
    generate_all_plots,
    load_metrics,
    plot_learning_curves,
    plot_td_loss,
    plot_reg_loss,
    plot_epsilon,
    plot_effective_rank,
    plot_svd_spectrum,
    plot_wall_time,
    plot_sample_efficiency,
    write_summary,
)


def _make_custom_dqn_records(n_episodes: int = 50, with_reg: bool = False):
    """Generate synthetic custom DQN metrics."""
    records = []
    for i in range(1, n_episodes + 1):
        entry = {
            "timestep": i * 20,
            "episode": i,
            "episode_return": 50.0 + 10 * (i / n_episodes) + ((-1) ** i) * 5,
            "episode_wall_time": 0.1,
            "cumulative_wall_time": i * 0.1,
        }
        if i > 10:
            entry["td_loss"] = max(0.01, 10.0 / i)
            entry["reg_loss"] = 0.5 / i if with_reg else 0.0
            entry["total_loss"] = entry["td_loss"] + entry["reg_loss"]
            entry["epsilon"] = max(0.05, 1.0 - i * 0.02)
        if i % 10 == 0:
            sv = [10.0 / (j + 1) for j in range(8)]
            if with_reg:
                sv = [sv[0], sv[1]] + [x * 0.1 for x in sv[2:]]
            entry["singular_values"] = sv
            entry["effective_rank"] = 3.5 if with_reg else 6.2
            entry["rank_above_threshold"] = 3 if with_reg else 7
        records.append(entry)
    return records


def _make_sb3_records(n_episodes: int = 50):
    """Generate synthetic SB3 metrics."""
    return [
        {
            "timestep": i * 20,
            "episode_return": 40.0 + 15 * (i / n_episodes) + ((-1) ** i) * 7,
            "episode_length": 20,
            "episode_wall_time": 0.05,
            "cumulative_wall_time": i * 0.05,
        }
        for i in range(1, n_episodes + 1)
    ]


@pytest.fixture
def comparison_dir(tmp_path):
    """Create a synthetic comparison output directory."""
    data = {
        "sb3_ppo": _make_sb3_records(),
        "sb3_dqn": _make_sb3_records(),
        "custom_dqn_noreg": _make_custom_dqn_records(with_reg=False),
        "custom_dqn_reg": _make_custom_dqn_records(with_reg=True),
    }
    for name, records in data.items():
        sub = tmp_path / name
        sub.mkdir()
        with open(sub / "metrics.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    run_config = {
        "num_machines": 5,
        "topology": "ring",
        "wall_times": {
            "sb3_ppo": 12.3,
            "sb3_dqn": 8.7,
            "custom_dqn_noreg": 15.1,
            "custom_dqn_reg": 16.4,
        },
    }
    with open(tmp_path / "run_config.json", "w") as f:
        json.dump(run_config, f)

    return tmp_path


def test_load_metrics(comparison_dir):
    metrics = load_metrics(comparison_dir)
    assert set(metrics.keys()) == {"sb3_ppo", "sb3_dqn", "custom_dqn_noreg", "custom_dqn_reg"}
    assert len(metrics["sb3_ppo"]) == 50


def test_plot_learning_curves(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "learning_curves.png"
    plot_learning_curves(metrics, out, smoothing=5)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_td_loss(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "td_loss.png"
    plot_td_loss(metrics, out)
    assert out.exists()


def test_plot_reg_loss(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "reg_loss.png"
    plot_reg_loss(metrics, out)
    assert out.exists()


def test_plot_epsilon(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "epsilon_schedule.png"
    plot_epsilon(metrics, out)
    assert out.exists()


def test_plot_effective_rank(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "effective_rank.png"
    plot_effective_rank(metrics, out)
    assert out.exists()


def test_plot_svd_spectrum(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "singular_value_spectrum.png"
    plot_svd_spectrum(metrics, out)
    assert out.exists()


def test_plot_wall_time(comparison_dir):
    from plot_results import load_run_config
    run_config = load_run_config(comparison_dir)
    out = comparison_dir / "wall_time.png"
    plot_wall_time(run_config, out)
    assert out.exists()


def test_plot_sample_efficiency(comparison_dir):
    metrics = load_metrics(comparison_dir)
    out = comparison_dir / "sample_efficiency.png"
    stats = plot_sample_efficiency(metrics, out)
    assert out.exists()
    assert "sb3_ppo" in stats


def test_write_summary(comparison_dir):
    metrics = load_metrics(comparison_dir)
    from plot_results import load_run_config
    run_config = load_run_config(comparison_dir)
    out = comparison_dir / "summary.json"
    write_summary(metrics, run_config, out)
    assert out.exists()
    summary = json.loads(out.read_text())
    assert "custom_dqn_reg" in summary
    assert "final_mean_return" in summary["custom_dqn_reg"]
    assert "auc" in summary["custom_dqn_reg"]


def test_generate_all_plots(comparison_dir):
    generate_all_plots(comparison_dir, smoothing=5)
    expected_files = [
        "learning_curves.png", "td_loss.png", "reg_loss.png",
        "epsilon_schedule.png", "effective_rank.png",
        "singular_value_spectrum.png", "wall_time.png",
        "sample_efficiency.png", "summary.json",
    ]
    for f in expected_files:
        assert (comparison_dir / f).exists(), f"Missing: {f}"


def test_graceful_with_missing_data(tmp_path):
    """Only SB3 data — custom DQN plots should skip gracefully."""
    sub = tmp_path / "sb3_ppo"
    sub.mkdir()
    records = _make_sb3_records(20)
    with open(sub / "metrics.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    generate_all_plots(tmp_path, smoothing=3)
    assert (tmp_path / "learning_curves.png").exists()
    # Custom-DQN-only plots should not be created
    assert not (tmp_path / "effective_rank.png").exists()
