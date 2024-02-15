import csv
import logging
import os
import re
import warnings
from functools import partial

import cv2
import dreamerv3
import jsonlines
import numpy as np
import ruamel.yaml as yaml
from carl.envs.carl_env import CARLEnv
from dreamerv3 import embodied
from dreamerv3.embodied.core.logger import _encode_gif

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    create_wrapped_carl_env,
)

logging.captureWarnings(True)
os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def generate_cartpole_length_envs(config):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    assert task == "classic_cartpole", task
    assert _TASK2CONTEXTS[task][1]["context"] == "length"
    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = []

    for v1 in _TASK2CONTEXTS[task][1]["interpolate_double"]:
        c = env_cls.get_default_context()
        if (
            config.env.carl.context == "default"
            or config.env.carl.context == "single_0"
        ) and v1 != c["length"]:
            mode = "extrapolate"
        else:
            mode = "interpolate"
        c["length"] = v1
        contexts.append({"context": c, "mode": mode})

    for v1 in _TASK2CONTEXTS[task][1]["extrapolate_double"]:
        c = env_cls.get_default_context()
        c["length"] = v1
        mode = "extrapolate"
        contexts.append({"context": c, "mode": mode})

    for context_info in contexts:
        envs = []
        ctor = lambda: create_wrapped_carl_env(
            env_cls, contexts={0: context_info["context"]}, config=config
        )
        ctor = partial(embodied.Parallel, ctor, "process")
        envs = [ctor()]
        yield embodied.BatchEnv(envs, parallel=True), context_info


def record_dream(agent, env, args, ctx_info, logdir):
    report = None

    def per_episode(ep):
        nonlocal agent, report
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.policy_devices)
        report = agent.report(jax_batch)
        video = report["openl_image"]
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
        encoded_img_str = _encode_gif(video, 30)
        l = ctx_info["context"]["length"]
        mode = ctx_info["mode"]
        fname = f"{mode}_length_{l:0.2f}"
        with open(logdir / f"{fname}.gif", "wb") as f:
            f.write(encoded_img_str)
        video = video[:10]
        # stack the video frames horizontally
        video = np.hstack(video)
        # draw a rectange around first 64 *5 pixels horizontally and 192 pixels vertically
        cv2.rectangle(video, (0, 0), (64 * 5, 192), (0, 0, 0), 2)
        # save the
        cv2.imwrite(str(logdir / f"{fname}.png"), video)

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=1)


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
    for env, ctx_info in generate_cartpole_length_envs(config):
        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )
        record_dream(agent, env, args, ctx_info, logdir)
        env.close()


if __name__ == "__main__":

    main()
