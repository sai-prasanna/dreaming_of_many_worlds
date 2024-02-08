import itertools
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
    "dmc_walker": [
        {
            "context": "gravity",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
    ],
    "dmc_quadruped": [
        {
            "context": "gravity",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
    ],
    "brax_ant": [
        {
            "context": "gravity",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
    ],
    "brax_halfcheetah": [
        {
            "context": "gravity",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
    ],
    "classic_cartpole": [
        {
            "context": "gravity",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
        {
            "context": "masspole",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
        {
            "context": "length",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
    ],
    "classic_pendulum": [
        {
            "context": "gravity",
            "interpolate": [[0.5, 1.5]],
            "extrapolate": [[0.1, 0.4], [1.6, 2.0]],
        },
    ],
}

_TASK2ENV = {
    "dmc_walker": CARLDmcWalkerEnv,
    "dmc_quadruped": CARLDmcQuadrupedEnv,
    "brax_ant": CARLBraxAnt,
    "brax_halfcheetah": CARLBraxHalfcheetah,
    "classic_pendulum": CARLPendulum,
    "classic_cartpole": CARLCartPole,
}


class ResizeImage(embodied.wrappers.ResizeImage):
    """Change interpolation to BILINEAR"""

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.BILINEAR)
        image = np.array(image)
        return image


def make_carl_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite

    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = {}

    if config.env.carl.context == "default":
        contexts = {0: env_cls.get_default_context()}
    elif "single" in config.env.carl.context:
        index = int(config.env.carl.context.split("_")[-1])
        context_name = _TASK2CONTEXTS[task][index]["context"]
        ctx_default_val = env_cls.get_default_context()[context_name]
        # since there is only one context we can only sample independent
        train_ranges = _TASK2CONTEXTS[task][index]["interpolate"]
        ctx_dists = [
            UniformFloatContextFeature(
                context_name, l * ctx_default_val, u * ctx_default_val
            )
            for l, u in train_ranges
        ]
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    elif "double_box" in config.env.carl.context:
        ctx_dists = []
        for index in range(2):
            context_name = _TASK2CONTEXTS[task][index]["context"]
            ctx_default_val = env_cls.get_default_context()[context_name]
            # since there is only one context we can only sample independent
            train_ranges = _TASK2CONTEXTS[task][index]["interpolate"]
            ctx_dists += [
                UniformFloatContextFeature(
                    context_name, l * ctx_default_val, u * ctx_default_val
                )
                for l, u in train_ranges
            ]
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    return create_wrapped_carl_env(env_cls, contexts, config)


def gen_carl_val_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]
    eval_distribution = overrides["eval_distribution"]
    contexts = []
    if config.env.carl.context == "default":
        if eval_distribution == "interpolate":
            contexts = [env_cls.get_default_context()]
        elif eval_distribution == "extrapolate":
            for i in range(len(_TASK2CONTEXTS[task])):
                context_name = _TASK2CONTEXTS[task][i]["context"]
                context_default = env_cls.get_default_context()[context_name]
                num_samples = 10 // len(_TASK2CONTEXTS[task][i]["extrapolate"])
                for l, u in _TASK2CONTEXTS[task][i]["extrapolate"]:
                    l, u = context_default * l, context_default * u
                    values = np.linspace(l, u, num_samples)
                    for v in values:
                        c = env_cls.get_default_context()
                        c[context_name] = v
                        contexts.append(c)
                num_samples = 10 // len(_TASK2CONTEXTS[task][i]["interpolate"])
                for l, u in _TASK2CONTEXTS[task][i]["interpolate"]:
                    l, u = context_default * l, context_default * u
                    values = np.linspace(l, u, num_samples)
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
            context_name = _TASK2CONTEXTS[task][index]["context"]
            context_default = env_cls.get_default_context()[context_name]
            # since there is only one context we can only sample independent
            num_samples = 10 // len(_TASK2CONTEXTS[task][index]["interpolate"])
            for l, u in _TASK2CONTEXTS[task][index]["interpolate"]:
                l, u = context_default * l, context_default * u
                values = np.linspace(l, u, num_samples)
                for v in values:
                    c = env_cls.get_default_context()
                    c[context_name] = v
                    contexts.append(c)
        elif eval_distribution == "extrapolate":
            assert "single" in config.env.carl.context
            index = int(config.env.carl.context.split("_")[-1])
            context_name = _TASK2CONTEXTS[task][index]["context"]
            context_default = env_cls.get_default_context()[context_name]
            num_samples = 10 // len(_TASK2CONTEXTS[task][index]["extrapolate"])
            for l, u in _TASK2CONTEXTS[task][index]["extrapolate"]:
                l, u = context_default * l, context_default * u
                values = np.linspace(l, u, num_samples)
                for v in values:
                    c = env_cls.get_default_context()
                    c[context_name] = v
                    contexts.append(c)

        else:
            raise NotImplementedError(
                f"Evaluation distribution {eval_distribution} not implemented."
            )
    elif "double_box" in config.env.carl.context:
        ctx_0_name = _TASK2CONTEXTS[task][0]["context"]
        ctx_0_default = env_cls.get_default_context()[ctx_0_name]
        ctx_0_interpolate_values = []
        for l, u in _TASK2CONTEXTS[task][0]["interpolate"]:
            l, u = ctx_0_default * l, ctx_0_default * u
            values = np.linspace(l, u, 3)
            ctx_0_interpolate_values += list(values)
        ctx_0_extrapolate_values = []
        for l, u in _TASK2CONTEXTS[task][0]["extrapolate"]:
            l, u = ctx_0_default * l, ctx_0_default * u
            values = np.linspace(l, u, 3)
            ctx_0_extrapolate_values += list(values)
        ctx_1_name = _TASK2CONTEXTS[task][1]["context"]
        ctx_1_default = env_cls.get_default_context()[ctx_1_name]
        ctx_1_interpolate_values = []
        for l, u in _TASK2CONTEXTS[task][1]["interpolate"]:
            l, u = ctx_1_default * l, ctx_1_default * u
            values = np.linspace(l, u, 3)
            ctx_1_interpolate_values += list(values)
        ctx_1_extrapolate_values = []
        for l, u in _TASK2CONTEXTS[task][1]["extrapolate"]:
            l, u = ctx_1_default * l, ctx_1_default * u
            values = np.linspace(l, u, 3)
            ctx_1_extrapolate_values += list(values)

        if eval_distribution == "interpolate":
            for v0, v1 in itertools.product(
                ctx_0_interpolate_values, ctx_1_interpolate_values
            ):
                c = env_cls.get_default_context()
                c[ctx_0_name] = v0
                c[ctx_1_name] = v1
                contexts.append(c)
        elif eval_distribution == "extrapolate":
            for v0, v1 in itertools.product(
                ctx_0_extrapolate_values, ctx_1_extrapolate_values
            ):
                c = env_cls.get_default_context()
                c[ctx_0_name] = v0
                c[ctx_1_name] = v1
                contexts.append(c)
        elif eval_distribution == "extrapolate_single":
            for v0, v1 in itertools.product(
                ctx_0_interpolate_values, ctx_1_extrapolate_values
            ):
                c = env_cls.get_default_context()
                c[ctx_0_name] = v0
                c[ctx_1_name] = v1
                contexts.append(c)
            for v0, v1 in itertools.product(
                ctx_0_extrapolate_values, ctx_1_interpolate_values
            ):
                c = env_cls.get_default_context()
                c[ctx_0_name] = v0
                c[ctx_1_name] = v1
                contexts.append(c)
        else:
            raise NotImplementedError(
                f"Evaluation distribution {eval_distribution} not implemented."
            )
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")
    for c in contexts:
        default_context = env_cls.get_default_context()
        context_info = {
            "context": c,
            "changed": [
                item["context"]
                for item in _TASK2CONTEXTS[task]
                if c[item["context"]] != default_context[item["context"]]
            ],
        }  # The context info is the context values for each
        # context feature which we potentially change
        ctors = []
        for index in range(config.envs.amount):
            ctor = lambda: create_wrapped_carl_env(
                env_cls, contexts={0: c}, config=config
            )
            if config.envs.parallel != "none":
                ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
            if config.envs.restart:
                ctor = bind(embodied.wrappers.RestartOnException, ctor)
            ctors.append(ctor)
        envs = [ctor() for ctor in ctors]
        yield embodied.BatchEnv(
            envs, parallel=(config.envs.parallel != "none")
        ), context_info


def create_wrapped_carl_env(env_cls, contexts, config):
    _, task = config.task.split("_", 1)
    env: CARLEnv = env_cls(
        contexts=contexts, obs_context_as_dict=False
    )  # Replace this with your Gym env.
    if "dmc" in task:
        env.env.render_mode = "rgb_array"
    if "classic" in task:
        env = TimeLimit(env, max_episode_steps=500)
    if task == "classic_cartpole":
        env.env.screen_width = 128
        env.env.screen_height = 128
    env.reset(seed=int(current_process().name.split("-")[-1]) + config.seed)
    env = from_gymnasium.FromGymnasium(env, obs_key="obs")
    env = embodied.core.wrappers.RenderImage(env, key="image")
    env = ResizeImage(env)
    return dreamerv3.wrap_env(env, config)
