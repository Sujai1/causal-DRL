"""Unified environment factory."""

from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import pyRDDLGym

from .wrappers import FlattenObsWrapper, SingleRebootActionWrapper
from ..graphs.causal_graph import CausalGraph
from ..graphs.extract_dbn import extract_causal_graph


def make_sysadmin_env(
    domain_path: Path,
    instance_path: Path,
    max_episode_steps: int = 100,
    seed: Optional[int] = None,
) -> Tuple[gym.Env, CausalGraph]:
    """Create wrapped SysAdmin environment and extract causal graph.

    Returns:
        env: Gymnasium env with flat obs, discrete actions, time limit.
        graph: CausalGraph with adjacency and k_global.
    """
    raw_env = pyRDDLGym.make(
        domain=str(domain_path), instance=str(instance_path)
    )

    graph = extract_causal_graph(domain_path, instance_path)

    env = FlattenObsWrapper(raw_env)
    env = SingleRebootActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if seed is not None:
        env.reset(seed=seed)

    return env, graph
