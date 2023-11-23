def main():
    import warnings

    import crafter
    import dreamerv3
    from carl.envs.dmc import CARLDmcWalkerEnv
    from dreamerv3 import embodied
    from gymnasium.wrappers import StepAPICompatibility

    from contextual_mbrl.dreamer import from_gymnasium

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["small"])
    config = config.update(
        {
            "logdir": "~/logdir/run1",
            "run.from_checkpoint": "~/logdir/run1/checkpoint.ckpt",
            "run.train_ratio": 64,
            "run.log_every": 30,  # Seconds
            "batch_size": 16,
            "jax.prealloc": False,
            "encoder.mlp_keys": "obs",
            "decoder.mlp_keys": "obs",
            "encoder.cnn_keys": "$^",
            "decoder.cnn_keys": "$^",
            "run.steps": 10000,
            # 'jax.platform': 'cpu',
        }
    )
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
        ],
    )
    ctx = CARLDmcWalkerEnv.get_default_context()
    ctx["gravity"] = -30
    env = CARLDmcWalkerEnv(
        {0: ctx}, obs_context_as_dict=False
    )  # Replace this with your Gym env.
    env.env.render_mode = "rgb_array"
    env = from_gymnasium.FromGymnasium(env, obs_key="obs")  # Or obs_key='vector'.
    print(env.obs_space)
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    embodied.run.eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
