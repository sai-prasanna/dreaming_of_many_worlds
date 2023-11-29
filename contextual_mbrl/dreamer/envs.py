from functools import partial as bind
from multiprocessing import current_process

import dreamerv3
from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from carl.envs.brax import CARLBraxAnt, CARLBraxHalfcheetah
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc import CARLDmcQuadrupedEnv, CARLDmcWalkerEnv
from carl.envs.gymnasium.classic_control import CARLCartPole, CARLPendulum
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gymnasium
from gymnasium.wrappers.time_limit import TimeLimit


def make_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != "none":
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(embodied.wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != "none"))


def make_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    if suite == "carl":
        return make_carl_env(config, **overrides)
    else:
        return dreamerv3.train.make_env(config, **overrides)


def make_carl_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    obs_key = "obs"

    task2env = {
        "dmc_walker": CARLDmcWalkerEnv,
        "dmc_quadruped": CARLDmcQuadrupedEnv,
        "brax_ant": CARLBraxAnt,
        "brax_halfcheetah": CARLBraxHalfcheetah,
        "classic_pendulum": CARLPendulum,
        "classic_cartpole": CARLCartPole,
    }

    env_cls: CARLEnv = task2env[task]
    contexts = {}
    if config.env.carl.context == "default":
        contexts = {0: env_cls.get_default_context()}
    elif config.env.carl.context == "vary_single":
        task2single_context = {
            "dmc_walker": "gravity",
            "dmc_quadruped": "gravity",
            "brax_ant": "gravity",
            "brax_halfcheetah": "gravity",
        }
        context_name = task2single_context[task]
        context_default = env_cls.get_default_context()[context_name]
        l, u = context_default / 2, context_default * 2
        sampler = ContextSampler(
            context_distributions=[UniformFloatContextFeature(context_name, l, u)],
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    env: CARLEnv = env_cls(
        contexts=contexts, obs_context_as_dict=False
    )  # Replace this with your Gym env.
    env.reset(seed=int(current_process().name.split("-")[-1]) + config.seed)
    if "dmc" in task:
        env.env.render_mode = "rgb_array"
    if "classic" in task:
        env = TimeLimit(env, max_episode_steps=500)

    env = from_gymnasium.FromGymnasium(env, obs_key=obs_key)

    if config.run.log_keys_video and config.run.log_keys_video[0] == "image":
        env = embodied.core.wrappers.RenderImage(env, key="image")
    return dreamerv3.wrap_env(env, config)
