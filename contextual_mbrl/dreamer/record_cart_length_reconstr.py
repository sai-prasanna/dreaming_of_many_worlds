import logging
import os
import warnings
from collections import defaultdict
from functools import partial

import cv2
import dreamerv3
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from carl.envs.carl_env import CARLEnv
from dreamerv3 import embodied, jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3.embodied.core.logger import _encode_gif

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    CARTPOLE_TRAIN_LENGTH_RANGE,
    create_wrapped_carl_env,
)
from contextual_mbrl.dreamer.record_cart_length_dreams import (
    _wrap_dream_agent,
    generate_cartpole_length_envs,
)

logging.captureWarnings(True)
os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def rollout_cart_length_reconst(agent, env, args, dream_agent_fn, episodes=10):
    record_ctx = []

    def per_episode(ep):
        nonlocal agent, record_ctx
        batch = {k: np.stack([v], 0) for k, v in ep.items()}

        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        ret_ctx, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        ret_ctx = agent._convert_mets(ret_ctx, agent.train_devices)

        # Remove normalization of length
        ctx = (ret_ctx["ctx"][0] + 1) / 2 * (
            CARTPOLE_TRAIN_LENGTH_RANGE[1] - CARTPOLE_TRAIN_LENGTH_RANGE[0]
        ) + CARTPOLE_TRAIN_LENGTH_RANGE[0]

        if np.where(ret_ctx["terminate"] > 0)[0].size > 0:
            terminate_idx = np.where(ret_ctx["terminate"] > 0)[0][0]
        else:
            terminate_idx = len(ret_ctx)
        record_ctx.append(ctx[:terminate_idx, 1].mean())

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=episodes)
    return record_ctx


def generate_cartpole_length_envs(config):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    assert task == "classic_cartpole", task
    assert _TASK2CONTEXTS[task][1]["context"] == "length"
    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = []

    for v1 in _TASK2CONTEXTS[task][1]["interpolate_single"]:
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

    for v1 in _TASK2CONTEXTS[task][1]["extrapolate_single"]:
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


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # create argparse with logdir
    parsed, other = embodied.Flags(logdir="", ctx_length=-1.0).parse_known()
    logdir = embodied.Path(parsed.logdir)
    ctx_length = parsed.ctx_length
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

    dream_agent_fn = None
    agent = None
    ctx_gt_pred = defaultdict(list)
    for env, ctx_info in generate_cartpole_length_envs(config):
        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent))
            dream_agent_fn = nj.jit(dream_agent_fn, device=agent.train_devices[0])
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )
        gt = ctx_info["context"]["length"]
        ctx_gt_pred[gt] = rollout_cart_length_reconst(
            agent, env, args, dream_agent_fn, episodes=5
        )
        env.close()
    # scatter plot the ground truth vs predicted length with sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # Make dataframe with gt vs predicted length, unroll the array
    df = pd.DataFrame(
        [
            {"gt": gt, "predicted length": pred}
            for gt, pred in ctx_gt_pred.items()
            for pred in pred
        ]
    )
    plt.figure(figsize=(20, 10))

    sns.scatterplot(data=df, x="gt", y="predicted length")
    # color the background in training range in green
    plt.axvspan(
        CARTPOLE_TRAIN_LENGTH_RANGE[0],
        CARTPOLE_TRAIN_LENGTH_RANGE[1],
        color="green",
        alpha=0.1,
    )

    plt.xlabel("ground truth length")
    plt.ylabel("predicted length")
    plt.title("Ground truth vs Predicted length")
    # set y range to start from 0
    plt.ylim(bottom=0)
    # show x-ticks
    plt.xticks(df["gt"].unique())
    plt.savefig(logdir / "gt_vs_pred_length.pdf")


if __name__ == "__main__":
    main()
