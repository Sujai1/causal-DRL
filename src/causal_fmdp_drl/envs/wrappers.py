"""Gymnasium wrappers for RDDL environments."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FlattenObsWrapper(gym.ObservationWrapper):
    """Flatten RDDL dict observations to a float32 Box.

    Extracts only state fluents matching a prefix pattern,
    converts bools to float32, and maintains consistent key
    ordering (sorted) across resets.
    """

    def __init__(self, env: gym.Env, fluent_prefix: str = "running"):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict), (
            f"Expected Dict observation space, got {type(obs_space)}"
        )

        self._obs_keys = sorted(
            k for k in obs_space.spaces if k.startswith(fluent_prefix)
        )
        assert len(self._obs_keys) > 0, (
            f"No fluents found with prefix '{fluent_prefix}'"
        )

        n = len(self._obs_keys)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        return np.array(
            [float(obs[k]) for k in self._obs_keys], dtype=np.float32
        )


class SingleRebootActionWrapper(gym.ActionWrapper):
    """Convert Discrete(m) action to RDDL reboot action dict.

    Action i maps to rebooting computer c{i+1} (0-indexed action
    to 1-indexed RDDL computer naming).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        act_space = env.action_space
        assert isinstance(act_space, spaces.Dict), (
            f"Expected Dict action space, got {type(act_space)}"
        )

        self._action_keys = sorted(act_space.spaces.keys())
        self.action_space = spaces.Discrete(len(self._action_keys))

    def action(self, action: int) -> dict:
        act_dict = {k: 0 for k in self._action_keys}
        act_dict[self._action_keys[action]] = 1
        return act_dict
