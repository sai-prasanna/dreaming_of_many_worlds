import csv
import logging
import os
import re
import warnings

import dreamerv3
import jsonlines
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied

from contextual_mbrl.dreamer.envs import gen_carl_val_envs

logging.captureWarnings(True)
os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def eval(agent, env, args, episodes=10):
    lengths = []
    rewards = []

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        lengths.append(length)
        rewards.append(score)

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=episodes)
    metrics = {
        "length": np.mean(lengths).astype(float),
        "length_std": np.std(lengths).astype(float),
        "length_min": np.min(lengths).astype(float),
        "length_max": np.max(lengths).astype(float),
        "return": np.mean(rewards).astype(float),
        "return_std": np.std(rewards).astype(float),
        "return_min": np.min(rewards).astype(float),
        "return_max": np.max(rewards).astype(float),
        "returns": rewards,
        "lengths": lengths,
    }

    return metrics


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # create argparse with logdir
    parsed, other = embodied.Flags(logdir="").parse_known()
    logdir = embodied.Path(parsed.logdir)
    # load the config from the logdir
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config = embodied.Config(config)
    # parse the overrides for eval
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})

    # Just load the step counter from the checkpoint, as there is
    # a circular dependency to load the agent.
    ckpt = embodied.Checkpoint()
    ckpt.step = embodied.Counter()
    ckpt.load(checkpoint, keys=["step"])
    step = ckpt._values["step"]

    agent = None
    returns = []
    lengths = []
    for env, ctx_info in gen_carl_val_envs(config):
        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )
        metrics = eval(agent, env, args, episodes=50)
        env.close()
        returns.extend(metrics["returns"])
        lengths.extend(metrics["lengths"])
        metrics["ctx"] = {**ctx_info}
        metrics["aggregated_context_metric"] = False
        metrics["checkpoint_step"] = int(step)

        # Write metrics to eval.jsonl
        with jsonlines.open(logdir / "eval.jsonl", mode="a") as writer:
            writer.write(metrics)


if __name__ == "__main__":
    main()
