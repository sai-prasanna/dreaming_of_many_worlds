import os
import re
import warnings

import numpy as np

os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo

import dreamerv3
import ruamel.yaml as yaml
from dreamerv3 import embodied

from contextual_mbrl.dreamer.envs import make_envs


def eval_only(agent, env, logger, args, prefix="eval", episodes=10):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        metrics.add({"length": length, "score": score}, prefix=f"{prefix}_stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=episodes)
    print(metrics.result(False))
    logger.add(metrics.result())
    logger.add(timer.stats(), prefix="timer")
    logger.write(fps=True)
    logger.write()


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    warnings.filterwarnings("once", ".*If you want to use these environments.*")

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

    loggers = [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, "eval_metrics.jsonl"),
        embodied.logger.TensorBoardOutput(logdir),
    ]
    if config.wandb.project != "":
        loggers.append(
            embodied.logger.WandBOutput(
                ".*",
                dict(
                    **config.wandb,
                    name=logdir.name,
                    config=dict(config),
                    resume=True,
                    dir=logdir,
                ),
            )
        )

    logger = embodied.Logger(step, loggers)

    for eval_dist, episodes in [("interpolate", 10), ("extrapolate", 100)]:
        env = make_envs(config, eval_distribution=eval_dist)

        agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )
        eval_only(agent, env, logger, args, prefix=eval_dist, episodes=episodes)


if __name__ == "__main__":
    main()
