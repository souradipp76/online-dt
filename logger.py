"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from datetime import datetime
import os
import utils
import wandb
import random

class Logger:
    def __init__(self, variant):

        self.log_path = self.create_log_path(variant)
        utils.mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")

        self.log_to_wandb = variant.get('log_to_wandb', False)
        env_name = variant['env']
        exp_prefix = 'gym-experiment'
        group_name = f'{exp_prefix}-{env_name}'
        exp_name = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

        run_id = variant["run_id"]
        if self.log_to_wandb:
            if run_id is not None:
                wandb.init(
                    name=exp_name,
                    group=group_name,
                    project='online-dt',
                    config=variant,
                    resume="must",
                    id=run_id
                )
            else:
                wandb.init(
                    name=exp_name,
                    group=group_name,
                    project='online-dt',
                    config=variant
                )

    def log_metrics(self, outputs, iter_num, total_transitions_sampled, writer):
        print("=" * 80)
        print(f"Iteration {iter_num}")
        for k, v in outputs.items():
            print(f"{k}: {v}")
            if writer:
                writer.add_scalar(k, v, iter_num)
                if k == "evaluation/return_mean_gm":
                    writer.add_scalar(
                        "evaluation/return_vs_samples",
                        v,
                        total_transitions_sampled,
                    )

        if self.log_to_wandb:
            log_dict = {}
            wandb.define_metric("training/iteration")
            wandb.define_metric("training/*", step_metric="training/iteration")
            log_dict["training/iteration"] = iter_num
            for k, v in outputs.items():
                log_dict[k] = v
                if k == "evaluation/return_mean_gm":
                    wandb.define_metric("evaluation/samples")
                    wandb.define_metric("evaluation/return_vs_samples", step_metric="evaluation/samples")
                    log_dict["evaluation/return_vs_samples"] = v
                    log_dict["evaluation/samples"] = total_transitions_sampled
            wandb.log(log_dict)        

    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        return f"{prefix}/{now}-{exp_name}"
