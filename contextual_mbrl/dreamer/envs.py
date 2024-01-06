from functools import partial as bind
from multiprocessing import current_process

import dreamerv3
import numpy as np
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


_TASK2CONTEXTS = {
    "dmc_walker": ["gravity"],
    "dmc_quadruped": ["gravity"],
    "brax_ant": ["gravity"],
    "brax_halfcheetah": ["gravity"],
    "classic_cartpole": ["gravity"],
    "classic_pendulum": ["gravity"],
}

_TASK2ENV = {
    "dmc_walker": CARLDmcWalkerEnv,
    "dmc_quadruped": CARLDmcQuadrupedEnv,
    "brax_ant": CARLBraxAnt,
    "brax_halfcheetah": CARLBraxHalfcheetah,
    "classic_pendulum": CARLPendulum,
    "classic_cartpole": CARLCartPole,
}


def make_carl_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    obs_key = "obs"

    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = {}

    if config.env.carl.context == "default":
        contexts = {0: env_cls.get_default_context()}
    elif "single" in config.env.carl.context:
        index = int(config.env.carl.context.split("_")[-1])
        context_name = _TASK2CONTEXTS[task][index]
        context_default = env_cls.get_default_context()[context_name]
        # since there is only one context we can only sample independent
        l, u = context_default * 0.5, context_default * 1.5
        sampler = ContextSampler(
            context_distributions=[UniformFloatContextFeature(context_name, l, u)],
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    env: CARLEnv = env_cls(
        contexts=contexts, obs_context_as_dict=False
    )  # Replace this with your Gym env.
    env.reset(seed=int(current_process().name.split("-")[-1]) + config.seed)
    if "dmc" in task:
        env.env.render_mode = "rgb_array"
    if "classic" in task:
        env = TimeLimit(env, max_episode_steps=500)

    env = from_gymnasium.FromGymnasium(env, obs_key=obs_key)
    env = embodied.core.wrappers.RenderImage(env, key="image")
    env = embodied.core.wrappers.ResizeImage(env)
    return dreamerv3.wrap_env(env, config)


def gen_carl_val_envs(config, **overrides):
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
    env_cls: CARLEnv = _TASK2ENV[task]
    eval_distribution = overrides["eval_distribution"]
    contexts = []
    if config.env.carl.context == "default":
        if eval_distribution == "interpolate":
            contexts = [env_cls.get_default_context()]
        elif eval_distribution == "extrapolate":
            context_default = env_cls.get_default_context()[context_name]
            l, u = context_default * 0.1, context_default * 2.0
            values = np.linspace(l, u, 10)
            for v in values:
                c = env_cls.get_default_context()
                c[context_name] = v
                contexts.append(c)
        else:
            raise NotImplementedError(
                f"Evaluation distribution {eval_distribution} not implemented."
            )
    elif "single" in config.env.carl.context:
        if eval_distribution == "interpolate":
            index = int(config.env.carl.context.split("_")[-1])
            context_name = _TASK2CONTEXTS[task][index]
            context_default = env_cls.get_default_context()[context_name]
            # since there is only one context we can only sample independent
            l, u = context_default * 0.5, context_default * 1.5
            for v in values:
                c = env_cls.get_default_context()
                c[context_name] = v
                contexts.append(c)
        elif eval_distribution == "extrapolate":
            assert "single" in config.env.carl.context
            index = int(config.env.carl.context.split("_")[-1])
            context_name = _TASK2CONTEXTS[task][index]
            context_default = env_cls.get_default_context()[context_name]
            l, u = context_default * 0.1, context_default * 0.9
            values = np.linspace(l, u, 5)
            for v in values:
                c = env_cls.get_default_context()
                c[context_name] = v
                contexts.append(c)
            l, u = context_default * 1.1, context_default * 2.0
            values = np.linspace(l, u, 5)
            for v in values:
                c = env_cls.get_default_context()
                c[context_name] = v
                contexts.append(c)
        else:
            raise NotImplementedError(
                f"Evaluation distribution {eval_distribution} not implemented."
            )
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    for c in contexts:
        default_context = env_cls.get_default_context()
        changed_context = {k: v for k, v in c.items() if default_context[k] != v}

        def make_eval_env():
            env: CARLEnv = env_cls(
                contexts={0: c}, obs_context_as_dict=False
            )  # Replace this with your Gym env.
            env.reset(seed=int(current_process().name.split("-")[-1]) + config.seed)
            print(int(current_process().name.split("-")[-1]))
            if "dmc" in task:
                env.env.render_mode = "rgb_array"
            if "classic" in task:
                env = TimeLimit(env, max_episode_steps=500)

            env = from_gymnasium.FromGymnasium(env, obs_key=obs_key)
            env = embodied.core.wrappers.RenderImage(env, key="image")
            env = embodied.core.wrappers.ResizeImage(env)
            return env

        ctors = []
        for index in range(config.envs.amount):
            ctor = lambda: make_eval_env()
            if config.envs.parallel != "none":
                ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
            if config.envs.restart:
                ctor = bind(embodied.wrappers.RestartOnException, ctor)
            ctors.append(ctor)
        envs = [ctor() for ctor in ctors]
        yield embodied.BatchEnv(
            envs, parallel=(config.envs.parallel != "none")
        ), changed_context
