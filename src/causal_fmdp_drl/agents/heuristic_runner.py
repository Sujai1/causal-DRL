"""Run heuristic (non-learning) baseline policies on SysAdmin environments."""

import json
import time
from pathlib import Path
from typing import Callable, Optional

from ..envs.make_env import make_sysadmin_env
from ..graphs.causal_graph import CausalGraph
from ..logging.jsonl_logger import JSONLLogger


def run_heuristic(
    policy_fn: Callable,
    policy_kwargs: dict,
    domain_path: Path,
    instance_path: Path,
    output_dir: Path,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    print_every: int = 10,
    graph: Optional[CausalGraph] = None,
) -> None:
    """Run a heuristic policy and log episode metrics.

    Args:
        policy_fn: Function (obs, **policy_kwargs) -> int action.
        policy_kwargs: Extra kwargs passed to policy_fn each step.
        domain_path: Path to RDDL domain file.
        instance_path: Path to RDDL instance file.
        output_dir: Directory to save metrics and graph.
        total_timesteps: Number of environment steps.
        max_episode_steps: TimeLimit wrapper horizon.
        seed: Random seed for env.
        print_every: Print progress every N episodes (0 = silent).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, graph = make_sysadmin_env(
        domain_path, instance_path,
        max_episode_steps=max_episode_steps, seed=seed, graph=graph,
    )

    with open(output_dir / "graph.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger = JSONLLogger(output_dir / "metrics.jsonl")

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_count = 0
    train_start = time.time()
    episode_start = time.time()

    for t in range(total_timesteps):
        action = policy_fn(obs, **policy_kwargs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

        if done:
            episode_count += 1
            now = time.time()

            logger.log({
                "timestep": t,
                "episode": episode_count,
                "episode_return": episode_return,
                "episode_wall_time": now - episode_start,
                "cumulative_wall_time": now - train_start,
            })

            if print_every > 0 and episode_count % print_every == 0:
                print(
                    f"  Episode {episode_count}: "
                    f"return={episode_return:.2f}"
                )

            obs, _ = env.reset()
            episode_return = 0.0
            episode_start = time.time()
        else:
            obs = next_obs

    logger.close()
    env.close()
