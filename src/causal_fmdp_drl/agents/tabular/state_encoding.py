"""State encoding utilities for tabular methods."""

import numpy as np


def obs_to_index(obs: np.ndarray) -> int:
    """Convert binary float observation to integer state index.

    The observation is a flat float32 array of 0.0/1.0 values representing
    binary state variables. We convert to an integer by treating as binary.

    Example: [1.0, 0.0, 1.0, 0.0, 0.0] -> binary 00101 -> int 5

    Args:
        obs: Float32 array of binary values (0.0 or 1.0).

    Returns:
        Integer state index.
    """
    bits = obs.astype(int)
    return int(sum(b << i for i, b in enumerate(bits)))


def index_to_obs(index: int, num_vars: int) -> np.ndarray:
    """Convert integer state index back to float observation.

    Args:
        index: Integer state index.
        num_vars: Number of state variables.

    Returns:
        Float32 array of binary values.
    """
    return np.array([(index >> i) & 1 for i in range(num_vars)], dtype=np.float32)


def check_tractable(num_vars: int, max_states: int = 50_000) -> bool:
    """Check if state space is tractable for tabular methods.

    Args:
        num_vars: Number of binary state variables.
        max_states: Maximum tractable state space size.

    Returns:
        True if tractable, False otherwise. Prints warning if not tractable.
    """
    num_states = 2 ** num_vars
    if num_states > max_states:
        print(
            f"WARNING: State space too large for tabular methods "
            f"(2^{num_vars} = {num_states:,} > {max_states:,}). "
            f"Skipping tabular baselines."
        )
        return False
    return True
