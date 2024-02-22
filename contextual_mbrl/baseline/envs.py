import itertools
from collections import deque

import numpy as np
from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from carl.envs.carl_env import CARLEnv
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from gymnasium.wrappers.time_limit import TimeLimit

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    NormalizeContextWrapper,
    RandomizedRoundRobinSelector,
)


def make_carl_train_env(config):
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


def gen_carl_val_envs(config):
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
        ctor = lambda: create_wrapped_carl_env(
            env_cls, contexts={0: context_info["context"]}, config=config
        )
        yield ctor, context_info


class ContextWrapper(ObservationWrapper):
    def __init__(self, env, use_context, num_stack=1):
        super().__init__(env)
        self.use_context = use_context
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        obs_low = env.observation_space["obs"].low
        obs_high = env.observation_space["obs"].high
        if self.use_context:
            # merge the "obs" and "context" spaces
            context_low = env.observation_space["context"].low
            context_high = env.observation_space["context"].high
            self.observation_space = spaces.Box(
                low=np.concatenate([obs_low] * num_stack + [context_low]),
                high=np.concatenate([obs_high] * num_stack + [context_high]),
                shape=(len(obs_low) * num_stack + len(context_low),),
                dtype=env.observation_space["obs"].dtype,
            )
            self.context = None
        else:
            self.observation_space = spaces.Box(
                low=np.concatenate([obs_low] * num_stack),
                high=np.concatenate([obs_high] * num_stack),
                shape=(len(obs_low) * num_stack,),
                dtype=env.observation_space["obs"].dtype,
            )

    def observation(self, observation):
        obs = np.concatenate(self.frames)
        if self.use_context:
            return np.concatenate([obs, self.context])
        else:
            return obs

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation["obs"])
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)
        if self.use_context:
            self.context = obs["context"]
        [self.frames.append(obs["obs"]) for _ in range(self.num_stack)]

        return self.observation(None), info


def create_wrapped_carl_env(env_cls: CARLEnv, contexts, config):
    _, task = config.task.split("_", 1)
    # Only the context features that might change in training or evaluation are # added to the observation space
    context_features = [o["context"] for o in _TASK2CONTEXTS[task]]
    seed = int(config.seed)
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

    env = ContextWrapper(
        env, use_context=config.use_context, num_stack=config.env.num_stack
    )

    return env
