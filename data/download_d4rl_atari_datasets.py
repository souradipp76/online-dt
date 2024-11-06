"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import pickle
import gym
import d4rl_atari

datasets = []

for env_name in ["breakout", "qbert", "pong"]:
    for dataset_type in ["mixed", "medium"]:
        name = f"{env_name}-{dataset_type}-v4"
        env = gym.make(name, stack = True)
        dataset = env.get_dataset()

        N = dataset["rewards"].shape[0]
        n = env.action_space.n

        paths = []
        done_steps = []
        for i in range(N):
            done_bool = bool(dataset["terminals"][i])
            if done_bool:
                done_steps.append(i)

        print(f"Number of paths: {len(done_steps)}")

        start_step = 0
        for done_step in done_steps:
            paths.append({
                "observations" : dataset["observations"][start_step : done_step + 1],
                "actions" : np.eye(n)[dataset["actions"][start_step : done_step + 1]],
                "rewards" : dataset["rewards"][start_step : done_step + 1],
                "terminals" : dataset["terminals"][start_step : done_step + 1]
            })
            start_step = done_step + 1

        returns = [np.sum(p["rewards"]) for p in paths]
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        print(f"Number of samples collected: {num_samples}")
        print(
            f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
        )

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(paths, f)