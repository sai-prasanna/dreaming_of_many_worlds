import os
import warnings

os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo
import dreamerv3
import ruamel.yaml as yaml
from dreamerv3 import embodied

from contextual_mbrl.dreamer.envs import make_envs


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    warnings.filterwarnings("once", ".*If you want to use these environments.*")

    parsed, other = embodied.Flags(configs=["defaults", "carl_dmc"]).parse_known()
    master_config = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "configs.yaml").read()
    )

    config = embodied.Config(master_config["defaults"])
    for name in parsed.configs:
        config = config.update(master_config[name])
    config = embodied.Flags(config).parse(other)
    logdir = embodied.Path(config.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    step = embodied.Counter()

    loggers = [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
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

    env = make_envs(config)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    embodied.run.train(agent, env, replay, logger, args)


if __name__ == "__main__":
    main()
