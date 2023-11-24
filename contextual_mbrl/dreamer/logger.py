import collections
import concurrent.futures
import datetime
import json
import os
import re
import time

import numpy as np


class WandBOutput:
    def __init__(self, pattern, logdir, config):
        self._pattern = re.compile(pattern)
        import wandb

        wandb.init(
            project=config.wandb.project,
            name=logdir.name,
            config=dict(config),
        )
        self._wandb = wandb

    def __call__(self, summaries):
        bystep = collections.defaultdict(dict)
        wandb = self._wandb
        for step, name, value in summaries:
            if len(value.shape) == 0 and self._pattern.search(name):
                bystep[step][name] = float(value)
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)
            elif len(value.shape) == 2:
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 3:
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 4:
                # Sanity check that the channeld dimension is last
                assert value.shape[3] in [1, 3, 4], f"Invalid shape: {value.shape}"
                value = np.transpose(value, [0, 3, 1, 2])
                # If the video is a float, convert it to uint8
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                bystep[step][name] = wandb.Video(value)

        for step, metrics in bystep.items():
            self._wandb.log(metrics, step=step)
