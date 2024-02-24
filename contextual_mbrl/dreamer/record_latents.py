import itertools
import logging
import os
import pickle
import warnings
from functools import partial

import dreamerv3
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied
from dreamerv3 import ninjax as nj

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    CARLEnv,
    create_wrapped_carl_env,
)

logging.captureWarnings(True)
os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def gen_carl_collect_latent_envs(config, **overrides):

    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]
    ctx_0_name = _TASK2CONTEXTS[task][0]["context"]
    ctx_1_name = _TASK2CONTEXTS[task][1]["context"]
    ctx_default = env_cls.get_default_context()
    ctx_0_default = ctx_default[ctx_0_name]
    ctx_1_default = ctx_default[ctx_1_name]
    contexts = []

    ctx_0_vals = (
        _TASK2CONTEXTS[task][0]["interpolate_double"]
        + _TASK2CONTEXTS[task][0]["extrapolate_double"]
    )
    assert ctx_0_default in ctx_0_vals
    ctx_1_vals = (
        _TASK2CONTEXTS[task][1]["interpolate_double"]
        + _TASK2CONTEXTS[task][1]["extrapolate_double"]
    )
    assert ctx_1_default in ctx_1_vals

    if config.env.carl.context in ["default", "single_0"]:
        for v in ctx_0_vals:
            c = env_cls.get_default_context()
            c[ctx_0_name] = v
            contexts.append({"context": c})
    if config.env.carl.context in ["default", "single_1"]:
        for v in ctx_1_vals:
            c = env_cls.get_default_context()
            if v == ctx_1_default and config.env.carl.context == "default":
                continue
            c[ctx_1_name] = v
            contexts.append({"context": c})
    if config.env.carl.context == "double_box":
        for v0, v1 in itertools.product(ctx_0_vals, ctx_1_vals):
            c = env_cls.get_default_context()
            c[ctx_0_name] = v0
            c[ctx_1_name] = v1
            contexts.append({"context": c})

    for context_info in contexts:
        ctor = lambda: create_wrapped_carl_env(
            env_cls, contexts={0: context_info["context"]}, config=config
        )
        ctor = partial(embodied.Parallel, ctor, "process")
        envs = [ctor()]
        yield embodied.BatchEnv(envs, parallel=True), context_info


def _wrap_dream_agent(agent):
    def gen_dream(data):
        n_start_imag, n_post, n_imag = 10, 5, 5
        data = agent.preprocess(data)
        wm = agent.wm
        # state = wm.initial(len(data["is_first"]))
        report = {}
        # report.update(wm.loss(data, state)[-1][-1])
        ep_len = data["is_first"].shape[1]
        if ep_len < (n_start_imag + n_imag):
            if ep_len >= 10:
                n_start_imag = 5
                n_post = 5
                n_imag = 5
            else:
                return report

        posterior_states, _ = wm.rssm.observe(
            wm.encoder(data)[:, :n_start_imag],
            data["action"][:, :n_start_imag],
            data["is_first"][:, :n_start_imag],
            dcontext=(
                data["context"][:, :n_start_imag] if wm.rssm._add_dcontext else None
            ),
        )

        # we can start imaginign from the last state of the posterior and all our actions
        start = {k: v[:, -1] for k, v in posterior_states.items()}
        posterior_states = (
            {**posterior_states, "context": data["context"][:, :n_start_imag]}
            if wm.rssm._add_dcontext
            else posterior_states
        )

        imagined_states = wm.rssm.imagine(
            data["action"][:, n_start_imag:],
            start,
            dcontext=(
                data["context"][:, n_start_imag:] if wm.rssm._add_dcontext else None
            ),
        )
        imagined_states = (
            {**imagined_states, "context": data["context"][:, n_start_imag:]}
            if wm.rssm._add_dcontext
            else imagined_states
        )
        # imagine_cont = wm.heads["cont"](imagined_states).mode()

        # terminate_idx = jnp.argwhere(imagine_cont == 0)
        # if len(terminate_idx) == 0:
        #     terminate_idx = len(imagine_cont)
        # else:
        #     terminate_idx = terminate_idx[0] + 1
        report["obs"] = data["obs"][0, n_start_imag - n_post : n_start_imag + n_imag]
        report["image"] = data["image"][
            0, n_start_imag - n_post : n_start_imag + n_imag
        ]

        report["posterior"] = jnp.concatenate(
            [
                posterior_states["deter"][0, n_start_imag - n_post : n_start_imag],
                jnp.reshape(
                    posterior_states["stoch"][0, n_start_imag - n_post : n_start_imag],
                    (
                        posterior_states["stoch"][
                            0, n_start_imag - n_post : n_start_imag
                        ].shape[0],
                        -1,
                    ),
                ),
            ],
            axis=1,
        )
        report["imagined"] = jnp.concatenate(
            [
                imagined_states["deter"][0, :n_imag],
                jnp.reshape(
                    imagined_states["stoch"][0, :n_imag],
                    (imagined_states["stoch"][0, :n_imag].shape[0], -1),
                ),
            ],
            axis=1,
        )

        return report

    return gen_dream


def collect_latents(agent, env, args, dream_agent_fn, episodes):
    report = []

    def per_episode(ep):
        nonlocal agent, report
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        r, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        if r:
            r = agent._convert_mets(r, agent.train_devices)
            report.append(r)

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=episodes)

    return report


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # create argparse with logdir
    parsed, other = embodied.Flags(logdir="", episodes=10).parse_known()
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

    dream_agent_fn = None
    agent = None
    ctx2latent = []
    suite, task = config.task.split("_", 1)
    ctx_0 = _TASK2CONTEXTS[task][0]["context"]
    ctx_1 = _TASK2CONTEXTS[task][1]["context"]

    for env, ctx_info in gen_carl_collect_latent_envs(config):

        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent))
            dream_agent_fn = nj.jit(dream_agent_fn, device=agent.train_devices[0])
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )
        logs = collect_latents(
            agent,
            env,
            args,
            dream_agent_fn,
            parsed.episodes,
        )
        ctx2latent.append(
            {
                "context": {
                    ctx_0: ctx_info["context"][ctx_0],
                    ctx_1: ctx_info["context"][ctx_1],
                },
                "context_order": [ctx_0, ctx_1],
                "episodes": logs,
            }
        )
        env.close()
    # save ctx2latent

    with (logdir / "ctx2latent_v1.pkl").open("wb") as f:
        pickle.dump(ctx2latent, f)


if __name__ == "__main__":
    main()
