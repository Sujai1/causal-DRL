"""Heuristic baseline policies for SysAdmin environments.

All policies return an integer action:
  - Action i (0-indexed) reboots machine i
  - Action num_machines = no-op

These are non-learning baselines for benchmarking RL agents.
"""

import numpy as np


def noop_policy(obs: np.ndarray, num_machines: int, **kw) -> int:
    """Always take no-op. Floor baseline."""
    return num_machines


def random_down_reboot_policy(
    obs: np.ndarray, num_machines: int, rng: np.random.Generator, **kw
) -> int:
    """Reboot a random down machine. No-op if all machines are running."""
    down = np.where(obs < 0.5)[0]
    if len(down) == 0:
        return num_machines
    return int(rng.choice(down))


def highest_degree_down_policy(
    obs: np.ndarray,
    num_machines: int,
    adj: np.ndarray,
    rng: np.random.Generator,
    **kw,
) -> int:
    """Reboot the down machine with highest degree (most connections).

    Proxies "protect hubs / high-impact nodes." Tie-break randomly.
    """
    down = np.where(obs < 0.5)[0]
    if len(down) == 0:
        return num_machines
    degrees = adj.sum(axis=1)
    degs = degrees[down]
    candidates = down[degs == degs.max()]
    return int(rng.choice(candidates))


def most_down_neighbors_policy(
    obs: np.ndarray,
    num_machines: int,
    adj: np.ndarray,
    rng: np.random.Generator,
    **kw,
) -> int:
    """Reboot the down machine with most currently-down neighbors.

    State-aware heuristic: prioritizes machines in failing neighborhoods.
    Tie-break by degree, then random.
    """
    down = np.where(obs < 0.5)[0]
    if len(down) == 0:
        return num_machines
    scores = np.array([
        np.sum(obs[np.where(adj[i] > 0)[0]] < 0.5) for i in down
    ])
    best = down[scores == scores.max()]
    if len(best) > 1:
        degrees = adj.sum(axis=1)
        degs = degrees[best]
        best = best[degs == degs.max()]
    return int(rng.choice(best))


def _expected_next_running(
    obs: np.ndarray,
    adj: np.ndarray,
    reboot_prob: float,
    reboot_action: int | None = None,
) -> np.ndarray:
    """Compute E[running_i'] for each machine given current obs and action.

    Uses the SysAdmin transition model:
      - Rebooted machine: P(up') = 1.0
      - Running, not rebooted: P(up') = 0.25 + 0.5*(1+running_nbrs)/(1+total_nbrs)
      - Down, not rebooted: P(up') = reboot_prob
    """
    n = len(obs)
    expected = np.zeros(n)
    for i in range(n):
        if reboot_action is not None and reboot_action == i:
            expected[i] = 1.0
        elif obs[i] > 0.5:  # running
            nbrs = np.where(adj[i] > 0)[0]
            running_nbrs = np.sum(obs[nbrs] > 0.5)
            total_nbrs = len(nbrs)
            expected[i] = 0.25 + 0.5 * (1 + running_nbrs) / (1 + total_nbrs)
        else:  # down
            expected[i] = reboot_prob
    return expected


def myopic_greedy_policy(
    obs: np.ndarray,
    num_machines: int,
    adj: np.ndarray,
    reboot_prob: float,
    reboot_penalty: float,
    **kw,
) -> int:
    """One-step lookahead: pick action maximizing E[reward at t+1].

    E[reward | action=noop] = sum_i E[running_i']
    E[reward | action=reboot(a)] = sum_i E[running_i'] - reboot_penalty

    Only considers rebooting down machines (rebooting an up machine
    costs 0.75 for negligible benefit).
    """
    # Expected next-step reward with no-op
    expected_noop = _expected_next_running(obs, adj, reboot_prob)
    best_value = expected_noop.sum()
    best_action = num_machines  # no-op

    # Only consider rebooting machines that are currently down
    down = np.where(obs < 0.5)[0]
    for a in down:
        expected_reboot = _expected_next_running(obs, adj, reboot_prob, reboot_action=a)
        value = expected_reboot.sum() - reboot_penalty
        if value > best_value:
            best_value = value
            best_action = int(a)

    return best_action
