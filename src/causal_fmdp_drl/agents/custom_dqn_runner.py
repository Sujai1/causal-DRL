"""Run custom DQN agent on wrapped RDDL environments."""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..envs.make_env import make_sysadmin_env
from ..logging.jsonl_logger import JSONLLogger
from .custom_dqn.agent import DQNAgent, DQNConfig

SVD_LOG_EVERY = 10  # Log SVD metrics every N episodes


def train_custom_dqn(
    domain_path: Path,
    instance_path: Path,
    output_dir: Path,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    lambda_reg: float = 0.0,
    dqn_config: Optional[DQNConfig] = None,
    print_every: int = 10,
    gamma: float = 0.95,
) -> None:
    """Train custom DQN agent and log metrics.

    Args:
        domain_path: Path to RDDL domain file.
        instance_path: Path to RDDL instance file.
        output_dir: Directory to save metrics, graph, and checkpoint.
        total_timesteps: Number of environment steps.
        max_episode_steps: TimeLimit wrapper horizon.
        seed: Random seed for numpy, torch, and env.
        lambda_reg: Regularization strength (0 = no regularization).
        dqn_config: Optional DQNConfig override. If None, uses defaults
            with the given lambda_reg.
        print_every: Print progress every N episodes (0 = silent).
        gamma: Discount factor.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, graph = make_sysadmin_env(
        domain_path, instance_path,
        max_episode_steps=max_episode_steps, seed=seed,
    )

    with open(output_dir / "graph.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    if dqn_config is None:
        dqn_config = DQNConfig(
            lambda_reg=lambda_reg,
            eps_decay_steps=int(total_timesteps * 0.1),
            gamma=gamma,
        )
    else:
        dqn_config.lambda_reg = lambda_reg
        dqn_config.eps_decay_steps = int(total_timesteps * 0.1)
        dqn_config.gamma = gamma

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=dqn_config,
        causal_graph=graph if lambda_reg > 0 else None,
    )

    logger = JSONLLogger(output_dir / "metrics.jsonl")

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_count = 0
    train_start = time.time()
    episode_start = time.time()

    for t in range(total_timesteps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        losses = agent.train_step(obs, action, reward, next_obs, float(done))
        episode_return += reward

        if done:
            episode_count += 1
            now = time.time()

            log_entry = {
                "timestep": t,
                "episode": episode_count,
                "episode_return": episode_return,
                "episode_wall_time": now - episode_start,
                "cumulative_wall_time": now - train_start,
                **losses,
            }

            # Log SVD metrics periodically
            if episode_count % SVD_LOG_EVERY == 0:
                svd_metrics = agent.compute_svd_metrics()
                log_entry.update(svd_metrics)

            logger.log(log_entry)

            if print_every > 0 and episode_count % print_every == 0:
                print(
                    f"  Episode {episode_count}: "
                    f"return={episode_return:.2f}, "
                    f"eps={agent.get_epsilon():.3f}"
                )

            obs, _ = env.reset()
            episode_return = 0.0
            episode_start = time.time()
        else:
            obs = next_obs

    torch.save(agent.q_net.state_dict(), output_dir / "q_net.pt")
    logger.close()
    env.close()
