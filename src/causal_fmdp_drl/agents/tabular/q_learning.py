"""Tabular Q-Learning agent."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TabularQConfig:
    """Configuration for tabular Q-learning.

    Default values follow standard literature (Sutton & Barto).
    """

    lr: float = 0.1  # Standard tabular learning rate
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.1  # Standard minimum exploration
    eps_decay_steps: int = 10_000  # Set by runner based on total_timesteps


class TabularQLearning:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: TabularQConfig,
    ):
        """Initialize Q-learning agent.

        Args:
            num_states: Size of discrete state space.
            num_actions: Number of actions.
            config: Hyperparameter configuration.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config

        # Initialize Q-table with zeros
        self.Q = np.zeros((num_states, num_actions), dtype=np.float64)

        self.step_count = 0

    def get_epsilon(self) -> float:
        """Linear epsilon decay."""
        progress = min(1.0, self.step_count / self.config.eps_decay_steps)
        return self.config.eps_start + progress * (
            self.config.eps_end - self.config.eps_start
        )

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Integer state index.
            eval_mode: If True, always act greedily.

        Returns:
            Selected action index.
        """
        if not eval_mode and np.random.random() < self.get_epsilon():
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.Q[state]))

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> dict:
        """Perform Q-learning update.

        Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state index.
            action: Action taken.
            reward: Reward received.
            next_state: Next state index.
            done: Whether episode terminated.

        Returns:
            Dict with update info (td_error).
        """
        self.step_count += 1

        # Compute TD target
        if done:
            target = reward
        else:
            target = reward + self.config.gamma * np.max(self.Q[next_state])

        # TD error
        td_error = target - self.Q[state, action]

        # Update Q-value
        self.Q[state, action] += self.config.lr * td_error

        return {"td_error": abs(td_error)}
