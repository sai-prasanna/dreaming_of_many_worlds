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
from carl.envs.dmc import CARLDmcQuadrupedEnv, CARLDmcWalkerEnv
from carl.envs.gymnasium.classic_control import CARLCartPole, CARLPendulum
from carl.utils.types import Context, Contexts
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gymnasium
from gymnasium import Wrapper, spaces
from gymnasium.wrappers.time_limit import TimeLimit

CARTPOLE_TRAIN_GRAVITY_RANGE_PCT = [0.5, 1.5]
CARTPOLE_TRAIN_LENGTH_RANGE_PCT = [0.7, 1.5]

_TASK2CONTEXTS = {
    "classic_cartpole": [
        {
            "context": "gravity",
            "train": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "interpolate": [0.55, 0.75, 0.95, 1.05, 1.25, 1.45],
            "extrapolate": [
                0.1,
                0.25,
                0.4,
                1.6,
                1.8,
                2.0,
            ],
        },
        {
            "context": "length",
            "train": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "interpolate": [0.75, 0.85, 0.95, 1.25, 1.35, 1.45],
            "extrapolate": [0.2, 0.4, 0.6, 1.6, 1.8, 2.0],
        },
    ],
}

_TASK2ENV = {
    "classic_cartpole": CARLCartPole,
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
            "gravity": CARTPOLE_TRAIN_GRAVITY_RANGE_PCT,
            "length": CARTPOLE_TRAIN_LENGTH_RANGE_PCT,
        }
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
        default_context = env_klass.get_default_context()
        self._train_low = np.array(
            [
                self._CLASS_TO_CTX_NORMALIZER[env_klass][k][0] * default_context[k]
                for k in env.obs_context_features
            ]
        )
        self._train_high = np.array(
            [
                self._CLASS_TO_CTX_NORMALIZER[env_klass][k][1] * default_context[k]
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
        ctx_default_val = env_cls.get_default_context()[context_name]
        # since there is only one context we can only sample independent
        train_vals = _TASK2CONTEXTS[task][index]["train"]
        contexts = {}
        for i, val in enumerate(train_vals):
            c = env_cls.get_default_context()
            c[context_name] = val * ctx_default_val
            contexts[i] = c
    elif "double_box" in config.env.carl.context:
        ctx_0_name = _TASK2CONTEXTS[task][0]["context"]
        ctx_1_name = _TASK2CONTEXTS[task][1]["context"]
        ctx_0_range = _TASK2CONTEXTS[task][0]["train"]
        ctx_1_range = _TASK2CONTEXTS[task][1]["train"]
        ctx_0_default = env_cls.get_default_context()[ctx_0_name]
        ctx_1_default = env_cls.get_default_context()[ctx_1_name]
        contexts = {}
        for i, (v0, v1) in enumerate(itertools.product(ctx_0_range, ctx_1_range)):
            c = env_cls.get_default_context()
            c[ctx_0_name] = v0 * ctx_0_default
            c[ctx_1_name] = v1 * ctx_1_default
            contexts[i] = c
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    return create_wrapped_carl_env(env_cls, contexts, config)


def gen_carl_val_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]

    ctx_0_name = _TASK2CONTEXTS[task][0]["context"]
    ctx_1_name = _TASK2CONTEXTS[task][1]["context"]
    ctx_0_range = (
        _TASK2CONTEXTS[task][0]["interpolate"]
        + _TASK2CONTEXTS[task][0]["extrapolate"]
        + [1.0]
    )
    ctx_1_range = (
        _TASK2CONTEXTS[task][1]["interpolate"]
        + _TASK2CONTEXTS[task][1]["extrapolate"]
        + [1.0]
    )
    ctx_0_default = env_cls.get_default_context()[ctx_0_name]
    ctx_1_default = env_cls.get_default_context()[ctx_1_name]
    contexts = {}
    for i, (v0, v1) in enumerate(itertools.product(ctx_0_range, ctx_1_range)):
        c = env_cls.get_default_context()
        c[ctx_0_name] = v0 * ctx_0_default
        c[ctx_1_name] = v1 * ctx_1_default
        contexts[i] = c

        changed = []
        if v0 != 1.0:
            changed.append(ctx_0_name)
        if v1 != 1.0:
            changed.append(ctx_1_name)
        context_info = {
            "context": c,
            "changed": changed,
        }
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


def create_wrapped_carl_env(env_cls: CARLEnv, contexts, config):
    _, task = config.task.split("_", 1)
    # Only the context features that might change in training or evaluation are # added to the observation space
    context_features = [o["context"] for o in _TASK2CONTEXTS[task]]
    seed = int(current_process().name.split("-")[-1]) + int(config.seed)
    env = env_cls(
        obs_context_as_dict=False,
        obs_context_features=context_features,
        context_selector=RandomizedRoundRobinSelector(seed, contexts),
    )  # Replace this with your Gym env.
    if "dmc" in task:
        env.env.render_mode = "rgb_array"
    if task == "classic_cartpole":
        env.env.screen_width = 128
        env.env.screen_height = 128
    env = NormalizeContextWrapper(env)
    if "classic" in task:
        env = TimeLimit(env, max_episode_steps=500)
    # reset once for paranoia
    env.reset(seed=seed)
    env = from_gymnasium.FromGymnasium(env, obs_key="obs")
    env = embodied.core.wrappers.RenderImage(env, key="image")
    env = ResizeImage(env)
    return dreamerv3.wrap_env(env, config)
