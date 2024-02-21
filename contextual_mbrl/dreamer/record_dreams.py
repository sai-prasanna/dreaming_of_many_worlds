import logging
import os
import warnings
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
    CARTPOLE_TRAIN_GRAVITY_RANGE,
    CARTPOLE_TRAIN_LENGTH_RANGE,
    create_wrapped_carl_env,
)

logging.captureWarnings(True)
os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def generate_envs(config, ctx_id):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    context_name = _TASK2CONTEXTS[task][ctx_id]["context"]
    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = []

    for v1 in _TASK2CONTEXTS[task][ctx_id]["interpolate_double"]:
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


def _wrap_dream_agent(agent):
    def gen_dream(data):
        data = agent.preprocess(data)
        wm = agent.wm
        state = wm.initial(len(data["is_first"]))
        report = {}
        report.update(wm.loss(data, state)[-1][-1])
        posterior_states, _ = wm.rssm.observe(
            wm.encoder(data)[:, :5],
            data["action"][:, :5],
            data["is_first"][:, :5],
            dcontext=data["context"][:6, :5] if wm.rssm._add_dcontext else None,
        )
        # we can start imaginign from the last state of the posterior and all our actions
        start = {k: v[:, -1] for k, v in posterior_states.items()}
        posterior_states = (
            {**posterior_states, "context": data["context"][:, :5]}
            if wm.rssm._add_dcontext
            else posterior_states
        )

        # when we decode posterior frames, we are just reconstructing as we
        # have the ground truth observations used for infering the latents
        # posterior_reconst = wm.heads["decoder"](posterior_states)
        # posterior_cont = wm.heads["cont"](posterior_states)

        posterior_reconst_5 = wm.heads["decoder"](posterior_states)
        posterior_cont_5 = wm.heads["cont"](posterior_states)

        imagined_states = wm.rssm.imagine(
            data["action"][:, 5:],
            start,
            dcontext=data["context"][:, 5:] if wm.rssm._add_dcontext else None,
        )
        imagined_states = (
            {**imagined_states, "context": data["context"][:, 5:]}
            if wm.rssm._add_dcontext
            else imagined_states
        )
        imagine_reconst = wm.heads["decoder"](imagined_states)
        imagine_cont = wm.heads["cont"](imagined_states)

        report["terminate"] = (
            1 - jnp.concatenate([posterior_cont_5.mode(), imagine_cont.mode()], 1)[0]
        )
        # report["terminate_post"] = 1 - posterior_cont.mode()[0]

        if "context" in data and "context" in posterior_reconst_5:
            model_ctx = jnp.concatenate(
                [
                    posterior_reconst_5["context"].mode(),
                    imagine_reconst["context"].mode(),
                ],
                1,
            )[
                ..., 1:
            ]  # pick only the length dimension of the context
            truth = data["context"][..., 1:]
            report["ctx"] = jnp.concatenate([truth, model_ctx], 2)
            # report["ctx_post"] = posterior_reconst["context"].mode()[..., 1:]
        truth = data["image"][:6].astype(jnp.float32)
        post_5_rest_imagined_reconst = jnp.concatenate(
            [
                posterior_reconst_5["image"].mode()[:, :5],
                imagine_reconst["image"].mode(),
            ],
            1,
        )
        error_1 = (post_5_rest_imagined_reconst - truth + 1) / 2
        # post_full_reconst = posterior_reconst["image"].mode()
        # error_2 = (post_full_reconst - truth + 1) / 2
        # video = jnp.concatenate(
        #     [truth, post_5_rest_imagined_reconst, error_1, post_full_reconst, error_2],
        #     2,
        # )
        video = jnp.concatenate(
            [truth, post_5_rest_imagined_reconst, error_1],
            2,
        )
        report["image"] = jaxutils.video_grid(video)
        return report

    return gen_dream


def record_dream(
    agent, env, args, ctx_info, logdir, dream_agent_fn, ctx_id, counterfactual_ctx, task
):
    report = None

    def per_episode(ep):
        nonlocal agent, report
        mode = ctx_info["mode"]
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        if counterfactual_ctx is not None:
            train_range = _TASK2CONTEXTS[task][ctx_id]["train_range"]
            batch["context"][..., ctx_id] = (counterfactual_ctx - train_range[0]) / (
                train_range[1] - train_range[0]
            ) * 2 - 1
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        report = agent._convert_mets(report, agent.train_devices)
        video = report["image"]

        video = np.clip(255 * video, 0, 255).astype(np.uint8)
        if "ctx" in report:
            # Remove normalization of length
            train_range = _TASK2CONTEXTS[task][ctx_id]["train_range"]
            ctx = (report["ctx"][0] + 1) / 2 * (
                train_range[1] - train_range[0]
            ) + train_range[0]
            for i in range(len(video)):
                video[i] = cv2.putText(
                    video[i],
                    f"{ctx[i][0]:.2f}",
                    (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                video[i] = cv2.putText(
                    video[i],
                    f"{ctx[i][1]:.2f}",
                    (10, 64 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                # video[i] = cv2.putText(
                #     video[i],
                #     f"{ctx[i][0] - ctx[i][1]:.2f}",
                #     (10, 64 * 2 + 15),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 0, 0),
                #     1,
                #     cv2.LINE_AA,
                # )
                # video[i] = cv2.putText(
                #     video[i],
                #     f"{post_ctx[i][0]:.2f}",
                #     (10, 64 * 3 + 15),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 0, 0),
                #     1,
                #     cv2.LINE_AA,
                # )
                # video[i] = cv2.putText(
                #     video[i],
                #     f"{ctx[i][0] - post_ctx[i][0]:.2f}",
                #     (10, 64 * 4 + 15),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 0, 0),
                #     1,
                #     cv2.LINE_AA,
                # )
        for i in range(len(video)):
            if report["terminate"][i] > 0:
                # draw a line at the right end of the image
                video[
                    i,
                    64:128,
                    -5:,
                ] = [255, 0, 0]
            # if report["terminate_post"][i] > 0:
            #     video[i, 64 * 3 :, -5:] = [255, 0, 0]

        encoded_img_str = _encode_gif(video, 30)
        context_name = _TASK2CONTEXTS[task][ctx_id]["context"]
        l = ctx_info["context"][context_name]

        fname = f"{mode}_length_{l:0.2f}"
        path = logdir / (
            f"dreams_{context_name}_{counterfactual_ctx}"
            if counterfactual_ctx is not None
            else "dreams_{context_name}"
        )
        path.mkdirs()
        with open(path / f"{fname}.gif", "wb") as f:
            f.write(encoded_img_str)
        # find the first terminate index
        if np.where(report["terminate"] > 0)[0].size > 0:
            terminate_idx = np.where(report["terminate"] > 0)[0][0]
        else:
            terminate_idx = len(video)
        video = video[: min(max(terminate_idx + 1, 100), len(video))]
        # stack the video frames horizontally
        video = np.hstack(video)
        # draw a rectange around first 64 *5 pixels horizontally and 192 pixels vertically
        cv2.rectangle(video, (0, 0), (64 * 5, 192), (0, 128, 0), 2)
        # save the
        cv2.imwrite(str(path / f"{fname}.png"), video[:, :, ::-1])

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
    parsed, other = embodied.Flags(
        logdir="", ctx_id=0, counterfactual_ctx=""
    ).parse_known()
    logdir = embodied.Path(parsed.logdir)
    counterfactual_ctx = (
        float(parsed.counterfactual_ctx) if parsed.counterfactual_ctx else None
    )
    ctx_id = parsed.ctx_id
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
    suite, task = config.task.split("_", 1)
    dream_agent_fn = None
    agent = None
    for env, ctx_info in generate_envs(config, ctx_id):
        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent))
            dream_agent_fn = nj.jit(dream_agent_fn, device=agent.train_devices[0])
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )

        record_dream(
            agent,
            env,
            args,
            ctx_info,
            logdir,
            dream_agent_fn,
            ctx_id,
            counterfactual_ctx,
            task,
        )
        env.close()


if __name__ == "__main__":
    # import sys

    # sys.argv[1:] = (
    #     "--logdir logs/carl_classic_cartpole_single_1_enc_img_dec_img_pgm_ctx_normalized/13/ --jax.train_devices 0 --jax.policy_devices 0".split(
    #         " "
    #     )
    # )
    main()
