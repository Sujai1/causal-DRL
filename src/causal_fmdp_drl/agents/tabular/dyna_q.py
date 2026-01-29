"""Dyna-Q agent: Q-learning with model-based planning."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .q_learning import TabularQLearning, TabularQConfig


@dataclass
class DynaQConfig(TabularQConfig):
    """Configuration for Dyna-Q.

    Extends TabularQConfig with planning-specific parameters.
    Default planning_steps follows Sutton & Barto recommendations.
    """

    planning_steps: int = 10  # Number of planning updates per real step


class DynaQ(TabularQLearning):
    """Dyna-Q agent: Q-learning with model-based planning.

    After each real transition, performs additional Q-learning updates
    using stored transitions (simulated experience from learned model).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: DynaQConfig,
    ):
        """Initialize Dyna-Q agent.

        Args:
            num_states: Size of discrete state space.
            num_actions: Number of actions.
            config: Hyperparameter configuration including planning_steps.
        """
        super().__init__(num_states, num_actions, config)
        self.config: DynaQConfig = config

        # Model: maps (state, action) -> list of observed (reward, next_state, done)
        self.model: Dict[Tuple[int, int], List[Tuple[float, int, bool]]] = {}

        # List of (state, action) pairs we've seen, for sampling during planning
        self.seen_pairs: List[Tuple[int, int]] = []

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> dict:
        """Perform Q-learning update, store transition, and do planning.

        Args:
            state: Current state index.
            action: Action taken.
            reward: Reward received.
            next_state: Next state index.
            done: Whether episode terminated.

        Returns:
            Dict with update info (td_error, planning_updates).
        """
        # Standard Q-learning update on real transition
        info = super().update(state, action, reward, next_state, done)

        # Store transition in model
        key = (state, action)
        if key not in self.model:
            self.model[key] = []
            self.seen_pairs.append(key)
        self.model[key].append((reward, next_state, done))

        # Planning: simulate experience from model
        planning_updates = 0
        for _ in range(self.config.planning_steps):
            if not self.seen_pairs:
                break

            # Sample random previously-seen (state, action)
            idx = np.random.randint(len(self.seen_pairs))
            plan_s, plan_a = self.seen_pairs[idx]

            # Sample random transition from that (state, action)
            transitions = self.model[(plan_s, plan_a)]
            trans_idx = np.random.randint(len(transitions))
            plan_r, plan_s_next, plan_done = transitions[trans_idx]

            # Q-learning update on simulated transition
            # (don't increment step_count for planning updates)
            if plan_done:
                target = plan_r
            else:
                target = plan_r + self.config.gamma * np.max(self.Q[plan_s_next])

            self.Q[plan_s, plan_a] += self.config.lr * (
                target - self.Q[plan_s, plan_a]
            )
            planning_updates += 1

        info["planning_updates"] = planning_updates
        return info
