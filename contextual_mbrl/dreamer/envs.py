import functools
import importlib
from functools import partial as bind
from typing import Any, Dict, Generic, TypeVar, Union, cast

import dreamerv3
import gymnasium
import numpy as np
from carl.context.context_space import (
    CategoricalContextFeature,
    NormalFloatContextFeature,
    UniformFloatContextFeature,
    UniformIntegerContextFeature,
)
from carl.context.sampler import ContextSampler
from carl.envs.brax import (
    CARLBraxAnt,
    CARLBraxHalfcheetah,
    CARLBraxHopper,
    CARLBraxHumanoid,
)
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc import (
    CARLDmcFingerEnv,
    CARLDmcFishEnv,
    CARLDmcQuadrupedEnv,
    CARLDmcWalkerEnv,
)
from dreamerv3 import embodied

U = TypeVar("U")
V = TypeVar("V")


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
    }

    env_cls: CARLEnv = task2env[task]
    contexts = {}
    if config.env.carl.context == "default":
        contexts = None
    elif config.env.carl.context == "vary_single":
        task2single_context = {
            "dmc_walker": "gravity",
            "dmc_quadruped": "gravity",
            "brax_ant": "gravity",
            "brax_halfcheetah": "gravity",
        }
        context_name = task2single_context[task]
        context_default = env_cls.get_default_context()[config.env.carl.context]
        l, u = context_default / 2, context_default * 2
        sampler = ContextSampler(
            context_distributions=[
                UniformFloatContextFeature(config.env.carl.context, l, u)
            ],
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    env = env_cls(
        contexts=contexts, obs_context_as_dict=False
    )  # Replace this with your Gym env.
    env.env.render_mode = "rgb_array"
    env = FromGymnasium(env, obs_key=obs_key)
    return dreamerv3.wrap_env(env, config)


class FromGymnasium(embodied.Env, Generic[U, V]):
    def __init__(
        self,
        env: Union[str, gymnasium.Env[U, V]],
        obs_key="image",
        act_key="action",
        **kwargs,
    ):
        if isinstance(env, str):
            self._env: gymnasium.Env[U, V] = gymnasium.make(
                env, render_mode="rgb_array", **kwargs
            )
        else:
            assert not kwargs, kwargs
            assert (
                env.render_mode == "rgb_array"
            ), f"render_mode must be rgb_array, got {self._env.render_mode}"
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            # cast is here to stop type checkers from complaining (we already check
            # that .spaces attr exists in __init__ as a proxy for the type check)
            obs_space = cast(gymnasium.spaces.Dict, self._env.observation_space)
            spaces = obs_space.spaces
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            act_space = cast(gymnasium.spaces.Dict, self._env.action_space)
            spaces = act_space.spaces
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = embodied.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            # we don't bother setting ._info here because it gets set below, once we
            # take the next .step()
            obs, _ = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            gymnasium_action = cast(V, self._unflatten(action))
        else:
            gymnasium_action = cast(V, action[self._act_key])
        obs, reward, terminated, truncated, self._info = self._env.step(
            gymnasium_action
        )
        self._done = terminated or truncated
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(self._info.get("is_terminal", self._done)),
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        np_obs: Dict[str, Any] = {k: np.asarray(v) for k, v in obs.items()}
        np_obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return np_obs

    def render(self):
        image = self._env.render()
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gymnasium.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)


# def make_carl_env(config, **overrides):
#     _, task = config.task.split("_", 1)
#     obs_key = "obs"
#     from carl.envs.dmc import (
#         CARLDmcFingerEnv,
#         CARLDmcFishEnv,
#         CARLDmcQuadrupedEnv,
#         CARLDmcWalkerEnv,
#     )

#     task2env = {
#         "dmc_walker": CARLDmcWalkerEnv,
#         "dmc_finger": CARLDmcFingerEnv,
#         "dmc_fish": CARLDmcFishEnv,
#         "dmc_quadruped": CARLDmcQuadrupedEnv,
#     }
#     env_ctor = task2env[task]
#     contexts = config.env.context.fixed.flat()

#     context_feat2ctor = {
#         "normal": NormalFloatContextFeature,
#         "uniform": UniformFloatContextFeature,
#         "uniform_int": UniformIntegerContextFeature,
#         "categorical": CategoricalContextFeature,
#     }

#     if config.env.context.sample:
#         # assuming all contexts can be sampled from normal distribution
#         context_distributions = []
#         for item in config.env.context.sample.items():
#             ctor_type = item.pop("type")
#             ctor = context_feat2ctor[ctor_type]
#             context_distributions.append(ctor(**item))
#         sampler = ContextSampler(
#             context_distributions=context_distributions,
#             context_space=env_ctor.get_context_space(),
#             seed=config.seed,
#         )
#         samples = sampler.sample_contexts(
#             n_contexts=10 ** len(context_distributions)
#         )
#         contexts.extend(samples)
#     env = env_ctor(
#         contexts=contexts, obs_context_as_dict=False
#     )  # Replace this with your Gym env.
#     env.env.render_mode = "rgb_array"
#     env = FromGymnasium(env, obs_key=obs_key)
#     return dreamerv3.wrap_env(env, config)
