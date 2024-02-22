import ruamel.yaml as yaml
from dreamerv3 import embodied
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from contextual_mbrl.baseline.envs import make_carl_train_env
from contextual_mbrl.baseline.model import create_model


def main():

    parsed, other = embodied.Flags(configs=["defaults"]).parse_known()
    master_config = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "configs.yaml").read()
    )
    config = embodied.Config(master_config["defaults"])
    for name in parsed.configs:
        config = config.update(master_config[name])
    config = embodied.Flags(config).parse(other)
    assert (
        config.algorithm != "recurrent_ppo" or config.env.num_stack == 1
    ), "Recurrent PPO does not require framestacking."

    logdir = embodied.Path(config.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    env_ctor = lambda: make_carl_train_env(config)
    logdir = embodied.Path(config.logdir)
    # Parallel environments

    if config.algorithm == "dqn":
        n_envs = 1
        timesteps = 5e4
    else:
        n_envs = 4
        timesteps = 1e5
    vec_env = make_vec_env(env_ctor, n_envs=n_envs)
    if config.algorithm == "recurrent_ppo":
        vec_env = VecNormalize(vec_env, gamma=0.9)
    model = create_model(config, vec_env)

    model.learn(total_timesteps=timesteps)
    model.save(str(logdir / "checkpoint.ckpt"))


if __name__ == "__main__":
    main()
