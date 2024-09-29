"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import collections
import pickle
import gym
import d4rl_atari

datasets = []

for env_name in ["breakout", "qbert", "pong"]:
    for dataset_type in ["mixed", "medium", "expert"]:
        name = f"{env_name}-{dataset_type}-v2"
        env = gym.make(env_name, stack = True)
        dataset = env.get_dataset()

        N = dataset["rewards"].shape[0]
        data_ = collections.defaultdict(list)
        n = env.action_space.n

        episode_step = 0
        paths = []
        for i in range(N):
            done_bool = bool(dataset["terminals"][i])
            final_timestep = episode_step == 1000 - 1
            for k in [
                "observations",
                "actions",
                "rewards",
                "terminals",
            ]:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                    if k == 'actions':
                        episode_data[k] = np.eye(n)[np.array(data_[k])]
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1

        returns = np.array([np.sum(p["rewards"]) for p in paths])
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        print(f"Number of samples collected: {num_samples}")
        print(
            f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
        )

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(paths, f)