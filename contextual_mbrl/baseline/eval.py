import jsonlines
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from contextual_mbrl.baseline.envs import gen_carl_val_envs
from contextual_mbrl.baseline.model import create_model


def main():

    # create argparse with logdir
    parsed, other = embodied.Flags(logdir="").parse_known()
    logdir = embodied.Path(parsed.logdir)
    # load the config from the logdir
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config = embodied.Config(config)
    # parse the overrides for eval
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint

    assert (
        config.algorithm != "recurrent_ppo" or config.env.num_stack == 1
    ), "Recurrent PPO does not require framestacking."

    agent = None
    for env, ctx_info in gen_carl_val_envs(config):
        vec_env = make_vec_env(env, n_envs=1)
        if config.algorithm == "recurrent_ppo":
            vec_env = VecNormalize(vec_env, gamma=0.9)
        if agent is None:
            agent = create_model(config, vec_env)
            agent = agent.load(str(logdir / "checkpoint.ckpt"), vec_env)
        agent.set_env(vec_env)
        returns, lengths = evaluate_policy(
            agent, vec_env, n_eval_episodes=50, return_episode_rewards=True
        )
        metrics = {
            "length": np.mean(lengths).astype(float),
            "length_std": np.std(lengths).astype(float),
            "length_min": np.min(lengths).astype(float),
            "length_max": np.max(lengths).astype(float),
            "return": np.mean(returns).astype(float),
            "return_std": np.std(returns).astype(float),
            "return_min": np.min(returns).astype(float),
            "return_max": np.max(returns).astype(float),
            "returns": returns,
            "lengths": lengths,
        }
        vec_env.close()
        metrics["ctx"] = {**ctx_info}

        # Write metrics to eval.jsonl
        with jsonlines.open(logdir / "eval.jsonl", mode="a") as writer:
            writer.write(metrics)


if __name__ == "__main__":
    main()
