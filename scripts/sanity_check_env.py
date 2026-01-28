"""Verify environment works end-to-end."""

from pathlib import Path

import numpy as np

from causal_fmdp_drl.envs.rddl.instance_generator import (
    generate_topology,
    write_sysadmin_instance,
)
from causal_fmdp_drl.envs.make_env import make_sysadmin_env


def main():
    m = 10
    topology = "erdos_renyi"
    er_prob = 0.2
    seed = 42
    H = 50
    num_episodes = 5

    artifacts_dir = Path("artifacts/rddl/sysadmin")
    domain_path = artifacts_dir / "domain.rddl"
    instances_dir = artifacts_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    adj = generate_topology(m, topology, seed=seed, er_prob=er_prob)
    instance_path = write_sysadmin_instance(
        adj, f"er_m{m}_p{er_prob}_s{seed}", instances_dir, horizon=H
    )

    env, graph = make_sysadmin_env(
        domain_path, instance_path, max_episode_steps=H, seed=seed
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"k_global: {graph.k_global}")
    print(f"K_causal(m={m}): {graph.K_causal(m)}")
    print(f"Graph density: {graph.density:.3f}")

    returns = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        assert obs.shape == env.observation_space.shape

        ep_return = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
        print(f"Episode {ep + 1}: return = {ep_return:.2f}")

    print(f"\nMean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print("Sanity check passed!")
    env.close()


if __name__ == "__main__":
    main()
