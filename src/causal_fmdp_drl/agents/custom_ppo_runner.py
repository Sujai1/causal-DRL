"""Run custom PPO agent on wrapped RDDL environments."""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..envs.make_env import make_sysadmin_env
from ..graphs.causal_graph import CausalGraph
from ..logging.jsonl_logger import JSONLLogger
from .custom_ppo.agent import PPOAgent, PPOConfig

SVD_LOG_EVERY = 10  # Log SVD metrics every N rollouts


def train_custom_ppo(
    domain_path: Path,
    instance_path: Path,
    output_dir: Path,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    ppo_config: Optional[PPOConfig] = None,
    print_every: int = 10,
    gamma: float = 0.95,
    use_layernorm: bool = False,
    n_steps: int = 2048,
    graph: Optional[CausalGraph] = None,
) -> None:
    """Train custom PPO agent and log metrics.

    Args:
        domain_path: Path to RDDL domain file.
        instance_path: Path to RDDL instance file.
        output_dir: Directory to save metrics, graph, and checkpoint.
        total_timesteps: Number of environment steps.
        max_episode_steps: TimeLimit wrapper horizon.
        seed: Random seed for numpy, torch, and env.
        ppo_config: Optional PPOConfig override.
        print_every: Print progress every N episodes (0 = silent).
        gamma: Discount factor.
        use_layernorm: Whether to use LayerNorm in networks.
        n_steps: Rollout length (overrides ppo_config.n_steps if ppo_config is None).
        graph: Optional pre-built causal graph (avoids XADD extraction).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, graph = make_sysadmin_env(
        domain_path, instance_path,
        max_episode_steps=max_episode_steps, seed=seed, graph=graph,
    )

    with open(output_dir / "graph.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    if ppo_config is None:
        ppo_config = PPOConfig(
            gamma=gamma,
            use_layernorm=use_layernorm,
            n_steps=n_steps,
        )
    else:
        ppo_config.gamma = gamma
        ppo_config.use_layernorm = use_layernorm
        ppo_config.n_steps = n_steps

    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config=ppo_config,
    )

    logger = JSONLLogger(output_dir / "metrics.jsonl")

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_count = 0
    episode_returns = []  # Track returns within current rollout
    train_start = time.time()
    rollout_count = 0
    total_steps = 0

    while total_steps < total_timesteps:
        # --- Collect rollout ---
        agent.buffer.reset()
        rollout_steps = min(ppo_config.n_steps, total_timesteps - total_steps)

        for step in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            agent.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                value=value,
            )

            episode_return += reward
            total_steps += 1
            agent.step_count += 1

            done = terminated or truncated
            if done:
                episode_count += 1
                episode_returns.append(episode_return)

                if print_every > 0 and episode_count % print_every == 0:
                    print(
                        f"  Episode {episode_count}: "
                        f"return={episode_return:.2f}, "
                        f"steps={total_steps}"
                    )

                obs, _ = env.reset()
                episode_return = 0.0
            else:
                obs = next_obs

        # --- Compute GAE ---
        # Get bootstrap value for the state after the last transition
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(ppo_config.device)
            last_value = agent.critic(obs_t).item()

        # Determine if the last transition in the buffer was terminal/truncated
        buf = agent.buffer
        buf_size = buf.size
        last_terminated = bool(buf.terminateds[buf_size - 1]) if buf_size > 0 else False
        last_truncated = bool(buf.truncateds[buf_size - 1]) if buf_size > 0 else False

        buf.compute_returns_and_advantages(
            last_value=last_value,
            last_terminated=last_terminated,
            last_truncated=last_truncated,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )

        # --- PPO update ---
        metrics = agent.update()
        rollout_count += 1
        now = time.time()

        # Log per-rollout metrics
        log_entry = {
            "timestep": total_steps,
            "rollout": rollout_count,
            "episodes_total": episode_count,
            "cumulative_wall_time": now - train_start,
        }

        if episode_returns:
            log_entry["mean_episode_return"] = float(np.mean(episode_returns))
            log_entry["min_episode_return"] = float(np.min(episode_returns))
            log_entry["max_episode_return"] = float(np.max(episode_returns))
            log_entry["episodes_in_rollout"] = len(episode_returns)

        log_entry.update(metrics)

        # Ensure all values are JSON-serializable (convert numpy scalars)
        for k, v in log_entry.items():
            if isinstance(v, (np.floating, np.integer)):
                log_entry[k] = v.item()

        # SVD and collapse diagnostics periodically
        if rollout_count % SVD_LOG_EVERY == 0:
            svd_metrics = agent.compute_svd_metrics()
            log_entry.update(svd_metrics)
            collapse_metrics = agent.compute_collapse_diagnostics()
            log_entry.update(collapse_metrics)

        logger.log(log_entry)
        episode_returns = []

    # Save checkpoints
    torch.save(agent.actor.state_dict(), output_dir / "actor.pt")
    torch.save(agent.critic.state_dict(), output_dir / "critic.pt")
    logger.close()
    env.close()
