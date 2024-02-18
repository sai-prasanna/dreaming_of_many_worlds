import copy
import itertools
import random
from functools import partial as bind
from multiprocessing import current_process
from typing import Tuple

import dreamerv3
import numpy as np
from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc import CARLDmcWalkerEnv
from carl.envs.gymnasium.classic_control import CARLCartPole, CARLPendulum
from carl.utils.types import Context, Contexts
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gymnasium
from gymnasium import Wrapper, spaces
from gymnasium.wrappers.time_limit import TimeLimit

CARTPOLE_TRAIN_GRAVITY_RANGE = [4.9, 14.70]
CARTPOLE_TRAIN_LENGTH_RANGE = [0.35, 0.75]

WALKER_TRAIN_GRAVITY_RANGE = [4.9, 14.70]
WALKER_TRAIN_ACTUATOR_STRENGTH_RANGE = [0.5, 1.5]

_TASK2CONTEXTS = {
    "classic_cartpole": [
        {
            "context": "gravity",
            "train_range": CARTPOLE_TRAIN_GRAVITY_RANGE,
            "interpolate_single": [4.9, 7.35, 9.8, 12.25, 14.7],
            "interpolate_double": [4.9, 9.8, 14.7],
            "extrapolate_single": [
                0.98,
                1.715,
                2.45,
                3.185,
                3.92,
                15.68,
                16.66,
                17.64,
                18.62,
                19.6,
            ],
            "extrapolate_double": [0.98, 2.45, 3.92, 15.68, 17.64, 19.6],
        },
        {
            "context": "length",
            "train_range": CARTPOLE_TRAIN_LENGTH_RANGE,
            "interpolate_single": [0.4, 0.5, 0.6, 0.7],
            "interpolate_double": [0.5, 0.7],
            "extrapolate_single": [
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ],
            "extrapolate_double": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        },
    ],
    "dmc_walker": [
        {
            "context": "gravity",
            "train_range": WALKER_TRAIN_GRAVITY_RANGE,
            "interpolate_single": [4.9, 7.35, 9.81, 12.25, 14.7],
            "interpolate_double": [4.9, 9.81, 14.7],
            "extrapolate_single": [
                0.98,
                1.715,
                2.45,
                3.185,
                3.92,
                15.68,
                16.66,
                17.64,
                18.62,
                19.6,
            ],
            "extrapolate_double": [0.98, 2.45, 3.92, 15.68, 17.64, 19.6],
        },
        {
            "context": "actuator_strength",
            "train_range": WALKER_TRAIN_ACTUATOR_STRENGTH_RANGE,
            "interpolate_single": [0.5, 0.75, 1.0, 1.25, 1.5],
            "interpolate_double": [0.5, 1.0, 1.5],
            "extrapolate_single": [0.1, 0.2, 0.3, 0.4, 1.6, 1.7, 1.8, 1.9, 2.0],
            "extrapolate_double": [0.1, 0.3, 1.6, 1.8, 2.0],
        },
    ],
}

_TASK2ENV = {
    "classic_cartpole": CARLCartPole,
    "dmc_walker": CARLDmcWalkerEnv,
}


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


class NormalizeContextWrapper(Wrapper):
    _CLASS_TO_CTX_NORMALIZER = {
        CARLCartPole: {
            "gravity": CARTPOLE_TRAIN_GRAVITY_RANGE,
            "length": CARTPOLE_TRAIN_LENGTH_RANGE,
        },
        CARLDmcWalkerEnv: {
            "gravity": WALKER_TRAIN_GRAVITY_RANGE,
            "actuator_strength": WALKER_TRAIN_ACTUATOR_STRENGTH_RANGE,
        },
    }

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, CARLEnv)
        assert "context" in env.observation_space.keys()
        assert isinstance(env.observation_space["context"], spaces.Box)
        assert not env.obs_context_as_dict
        # Normalize context space to [0, 1]
        self.observation_space["context"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space["context"].shape,
            dtype=np.float32,
        )
        env_klass = type(env)
        self._train_low = np.array(
            [
                self._CLASS_TO_CTX_NORMALIZER[env_klass][k][0]
                for k in env.obs_context_features
            ]
        )
        self._train_high = np.array(
            [
                self._CLASS_TO_CTX_NORMALIZER[env_klass][k][1]
                for k in env.obs_context_features
            ]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["context"] = self._normalize_context(obs["context"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["context"] = self._normalize_context(obs["context"])
        return obs, reward, terminated, truncated, info

    def _normalize_context(self, context):
        return (context - self._train_low) / (
            self._train_high - self._train_low
        ) * 2 - 1


class RandomizedRoundRobinSelector(AbstractSelector):
    """
    Round robin context selector.

    Iterate through all contexts and then start at the first again.
    """

    def __init__(self, seed: int, contexts: Contexts):
        super().__init__(contexts)
        self.rand = random.Random(seed)

    def _select(self) -> Tuple[Context, int]:
        if self.context_id is None:
            self.context_id = -1
        self.context_id = (self.context_id + 1) % len(self.contexts)
        if self.context_id == 0:
            self.rand.shuffle(self.contexts)
        context = self.contexts[self.contexts_keys[self.context_id]]
        return context, self.context_id


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
        # since there is only one context we can only sample independent
        train_range = _TASK2CONTEXTS[task][index]["train_range"]
        ctx_dists = [
            UniformFloatContextFeature(context_name, train_range[0], train_range[1])
        ]
        # All workers sample same contexts based on experiment seed
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
            train_range = _TASK2CONTEXTS[task][index]["train_range"]
            ctx_dists.append(
                UniformFloatContextFeature(context_name, train_range[0], train_range[1])
            )
        # All workers sample same contexts based on experiment seed
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    elif "specific" in config.env.carl.context:
        ctx_vals = {
            int(v.split("-")[0]): float(v.split("-")[-1])
            for v in config.env.carl.context.split("_")[1:]
        }
        specific_context = env_cls.get_default_context()
        for i, val in ctx_vals.items():
            context_name = _TASK2CONTEXTS[task][i]["context"]
            specific_context[context_name] = val
        contexts[0] = specific_context
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    return create_wrapped_carl_env(env_cls, contexts, config)


def gen_carl_val_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]
    ctx_0_name = _TASK2CONTEXTS[task][0]["context"]
    ctx_1_name = _TASK2CONTEXTS[task][1]["context"]
    ctx_default = env_cls.get_default_context()
    ctx_0_default = ctx_default[ctx_0_name]
    ctx_1_default = ctx_default[ctx_1_name]
    contexts = []

    if "specific" in config.env.carl.context:
        ctx_vals = {
            int(v.split("-")[0]): float(v.split("-")[-1])
            for v in config.env.carl.context.split("_")[1:]
        }
        c = env_cls.get_default_context()
        changed = []
        for i, val in ctx_vals.items():
            context_name = _TASK2CONTEXTS[task][i]["context"]
            c[context_name] = val
            changed.append(context_name)
        contexts.append({"context": c, "changed": changed})
    else:

        ctx_0_single = (
            _TASK2CONTEXTS[task][0]["interpolate_single"]
            + _TASK2CONTEXTS[task][0]["extrapolate_single"]
        )
        assert ctx_0_default in ctx_0_single
        for v0 in ctx_0_single:
            c = env_cls.get_default_context()
            c[ctx_0_name] = v0
            changed = []
            if v0 != ctx_0_default:
                changed.append(ctx_0_name)
            contexts.append({"context": c, "changed": changed})

        ctx_1_single = (
            _TASK2CONTEXTS[task][1]["interpolate_single"]
            + _TASK2CONTEXTS[task][1]["extrapolate_single"]
        )
        assert ctx_1_default in ctx_1_single
        for v1 in ctx_1_single:
            c = env_cls.get_default_context()
            c[ctx_1_name] = v1
            if v1 == ctx_1_default:
                continue
            contexts.append({"context": c, "changed": [ctx_1_name]})

        ctx_0_double = (
            _TASK2CONTEXTS[task][0]["interpolate_double"]
            + _TASK2CONTEXTS[task][0]["extrapolate_double"]
        )
        assert ctx_0_default in ctx_0_double
        ctx_1_double = (
            _TASK2CONTEXTS[task][1]["interpolate_double"]
            + _TASK2CONTEXTS[task][1]["extrapolate_double"]
        )
        assert ctx_1_default in ctx_1_double
        for v0, v1 in itertools.product(ctx_0_double, ctx_1_double):
            c = env_cls.get_default_context()
            c[ctx_0_name] = v0
            c[ctx_1_name] = v1
            # We make sure that default context is already covered in interpolate single case
            if v0 == ctx_0_default or v1 == ctx_1_default:
                continue
            contexts.append({"context": c, "changed": [ctx_0_name, ctx_1_name]})

    for context_info in contexts:
        ctors = []
        for index in range(config.envs.amount):
            ctor = lambda: create_wrapped_carl_env(
                env_cls, contexts={0: context_info["context"]}, config=config
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


def create_wrapped_carl_env(env_cls: CARLEnv, contexts, config):
    _, task = config.task.split("_", 1)
    # Only the context features that might change in training or evaluation are # added to the observation space
    context_features = [o["context"] for o in _TASK2CONTEXTS[task]]
    seed = int(current_process().name.split("-")[-1]) + int(config.seed)
    env = env_cls(
        obs_context_as_dict=False,
        obs_context_features=context_features,
        context_selector=RandomizedRoundRobinSelector(seed, contexts),
        kwargs={"render_mode": "rgb_array"},
    )  # Replace this with your Gym env.
    # reset once for paranoia
    env.reset(seed=seed)
    if "dmc" in task:
        env.env.render_mode = "rgb_array"
    if task == "classic_cartpole":
        env.env.screen_width = 128
        env.env.screen_height = 128
    env = NormalizeContextWrapper(env)
    if "classic" in task:
        env = TimeLimit(env, max_episode_steps=500)

    env = from_gymnasium.FromGymnasium(env, obs_key="obs")
    env = embodied.core.wrappers.RenderImage(env, key="image")
    env = ResizeImage(env)
    return dreamerv3.wrap_env(env, config)
