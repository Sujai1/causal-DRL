"""Run tabular RL agents on wrapped RDDL environments."""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..envs.make_env import make_sysadmin_env
from ..logging.jsonl_logger import JSONLLogger
from .tabular.state_encoding import obs_to_index, check_tractable
from .tabular.q_learning import TabularQLearning, TabularQConfig
from .tabular.dyna_q import DynaQ, DynaQConfig


def _run_tabular_training(
    agent,
    env,
    graph,
    output_dir: Path,
    total_timesteps: int,
    seed: int,
    print_every: int,
    agent_name: str,
) -> None:
    """Shared training loop for tabular agents.

    Args:
        agent: TabularQLearning or DynaQ agent.
        env: Wrapped gymnasium environment.
        graph: CausalGraph for logging.
        output_dir: Directory to save metrics and graph.
        total_timesteps: Number of environment steps.
        seed: Random seed.
        print_every: Print progress every N episodes (0 = silent).
        agent_name: Name for logging.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save graph for consistency with other baselines
    with open(output_dir / "graph.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger = JSONLLogger(output_dir / "metrics.jsonl")

    obs, _ = env.reset(seed=seed)
    state = obs_to_index(obs)
    episode_return = 0.0
    episode_count = 0
    train_start = time.time()
    episode_start = time.time()

    for t in range(total_timesteps):
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = obs_to_index(next_obs)

        agent.update(state, action, reward, next_state, done)
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
                "epsilon": agent.get_epsilon(),
            }
            logger.log(log_entry)

            if print_every > 0 and episode_count % print_every == 0:
                print(
                    f"  Episode {episode_count}: "
                    f"return={episode_return:.2f}, "
                    f"eps={agent.get_epsilon():.3f}"
                )

            obs, _ = env.reset()
            state = obs_to_index(obs)
            episode_return = 0.0
            episode_start = time.time()
        else:
            state = next_state

    # Save Q-table
    np.save(output_dir / "q_table.npy", agent.Q)

    logger.close()
    env.close()


def train_tabular_q(
    domain_path: Path,
    instance_path: Path,
    output_dir: Path,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    config: Optional[TabularQConfig] = None,
    print_every: int = 10,
    eps_decay_frac: float = 0.5,
) -> None:
    """Train tabular Q-learning agent and log metrics.

    Args:
        domain_path: Path to RDDL domain file.
        instance_path: Path to RDDL instance file.
        output_dir: Directory to save metrics, graph, and Q-table.
        total_timesteps: Number of environment steps.
        max_episode_steps: TimeLimit wrapper horizon.
        seed: Random seed.
        config: Optional TabularQConfig override.
        print_every: Print progress every N episodes (0 = silent).
        eps_decay_frac: Fraction of total_timesteps over which to decay epsilon.
    """
    np.random.seed(seed)

    env, graph = make_sysadmin_env(
        domain_path, instance_path, max_episode_steps=max_episode_steps, seed=seed
    )

    num_states = 2 ** env.observation_space.shape[0]
    num_actions = env.action_space.n

    if config is None:
        config = TabularQConfig(eps_decay_steps=int(total_timesteps * eps_decay_frac))
    else:
        config.eps_decay_steps = int(total_timesteps * eps_decay_frac)

    agent = TabularQLearning(num_states, num_actions, config)

    _run_tabular_training(
        agent=agent,
        env=env,
        graph=graph,
        output_dir=output_dir,
        total_timesteps=total_timesteps,
        seed=seed,
        print_every=print_every,
        agent_name="Tabular Q-Learning",
    )


def train_dyna_q(
    domain_path: Path,
    instance_path: Path,
    output_dir: Path,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 100,
    seed: int = 0,
    planning_steps: int = 10,
    config: Optional[DynaQConfig] = None,
    print_every: int = 10,
    eps_decay_frac: float = 0.5,
) -> None:
    """Train Dyna-Q agent and log metrics.

    Args:
        domain_path: Path to RDDL domain file.
        instance_path: Path to RDDL instance file.
        output_dir: Directory to save metrics, graph, and Q-table.
        total_timesteps: Number of environment steps.
        max_episode_steps: TimeLimit wrapper horizon.
        seed: Random seed.
        planning_steps: Number of planning updates per real step.
        config: Optional DynaQConfig override.
        print_every: Print progress every N episodes (0 = silent).
        eps_decay_frac: Fraction of total_timesteps over which to decay epsilon.
    """
    np.random.seed(seed)

    env, graph = make_sysadmin_env(
        domain_path, instance_path, max_episode_steps=max_episode_steps, seed=seed
    )

    num_states = 2 ** env.observation_space.shape[0]
    num_actions = env.action_space.n

    if config is None:
        config = DynaQConfig(
            eps_decay_steps=int(total_timesteps * eps_decay_frac),
            planning_steps=planning_steps,
        )
    else:
        config.eps_decay_steps = int(total_timesteps * eps_decay_frac)
        config.planning_steps = planning_steps

    agent = DynaQ(num_states, num_actions, config)

    _run_tabular_training(
        agent=agent,
        env=env,
        graph=graph,
        output_dir=output_dir,
        total_timesteps=total_timesteps,
        seed=seed,
        print_every=print_every,
        agent_name="Dyna-Q",
    )
