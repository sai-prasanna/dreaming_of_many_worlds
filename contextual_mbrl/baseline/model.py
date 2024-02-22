from typing import Callable, Union

from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3 import DQN, PPO, SAC
from torch import nn


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def create_model(config, env):
    if config.algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=2.3e-3,
            batch_size=256,
            buffer_size=100000,
            learning_starts=1000,
            gamma=0.99,
            target_update_interval=10,
            train_freq=256,
            gradient_steps=128,
            exploration_fraction=0.16,
            exploration_final_eps=0.04,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
        )
    elif config.algorithm == "ppo":
        if "cartpole" in config.task:
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=32,
                batch_size=256,
                gae_lambda=0.8,
                gamma=0.98,
                n_epochs=20,
                ent_coef=0.0,
                learning_rate=linear_schedule(0.001),
                clip_range=linear_schedule(0.2),
                verbose=1,
            )
        elif "pendulum" in config.task:
            model = PPO(
                "MlpPolicy",
                env,
                gamma=0.98,
                # Using https://proceedings.mlr.press/v164/raffin22a.html
                use_sde=True,
                sde_sample_freq=4,
                learning_rate=1e-3,
                verbose=1,
            )
        else:
            raise RuntimeError(f"Unknown task: {config.task}")
    elif config.algorithm == "recurrent_ppo":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=1024,
            gae_lambda=0.95,
            gamma=0.9,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=1e-3,
            clip_range=0.2,
            # use_sde=True,
            # sde_sample_freq=4,
            policy_kwargs=dict(
                ortho_init=False,
                activation_fn=nn.ReLU,
                lstm_hidden_size=64,
                enable_critic_lstm=True,
                net_arch=dict(pi=[64], vf=[64]),
            ),
            verbose=1,
        )
    elif config.algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
        )

    return model
