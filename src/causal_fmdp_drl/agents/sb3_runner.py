"""Run Stable-Baselines3 agents on wrapped RDDL environments."""

import json
import time
from pathlib import Path
from typing import Literal, Optional

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from ..envs.make_env import make_sysadmin_env
from ..graphs.causal_graph import CausalGraph
from ..logging.jsonl_logger import JSONLLogger


class MetricsCallback(BaseCallback):
    """Log episode returns to JSONL."""

    def __init__(self, logger: JSONLLogger):
        super().__init__()
        self._jsonl_logger = logger
        self._train_start = time.time()
        self._episode_start = time.time()

    def _on_step(self) -> bool:
        now = time.time()
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._jsonl_logger.log({
                    "timestep": self.num_timesteps,
                    "episode_return": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                    "episode_wall_time": now - self._episode_start,
                    "cumulative_wall_time": now - self._train_start,
                })
                self._episode_start = time.time()
        return True


def train_sb3(
    algo: Literal["dqn", "ppo"],
    domain_path: Path,
    instance_path: Path,
    output_dir: Path,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    graph: Optional[CausalGraph] = None,
    **algo_kwargs,
) -> None:
    """Train SB3 agent and log metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, graph = make_sysadmin_env(
        domain_path, instance_path, max_episode_steps=max_episode_steps, seed=seed,
        graph=graph,
    )

    with open(output_dir / "graph.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger = JSONLLogger(output_dir / "metrics.jsonl")
    callback = MetricsCallback(logger)

    algo_cls = {"dqn": DQN, "ppo": PPO}[algo]
    model = algo_cls("MlpPolicy", env, seed=seed, verbose=1, **algo_kwargs)

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(output_dir / f"{algo}_model")
    logger.close()
    env.close()
