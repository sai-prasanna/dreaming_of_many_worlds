#!/bin/bash

#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_asingle
#SBATCH --output logs/slurm/single_job.out
#SBATCH --error logs/slurm/single_job.err
#SBATCH --mem 64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2

echo "Workingdir: $PWD";
echo "Started at $(date)";

source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate c_mbrl

start=`date +%s`

python -m contextual_mbrl.dreamer.train --configs carl enc_obs_ctx_dec_obs --task carl_classic_cartpole --env.carl.context vary_single --seed 1337 --logdir logs/carl_classic_cartpole_vary_single_enc_obs_ctx_dec_obs/1337 --wandb.group carl_classic_cartpole_vary_single_enc_obs_ctx_dec_obs --jax.policy_devices 0 --jax.train_devices 1 --run.steps 25000

python -m contextual_mbrl.dreamer.train --configs carl enc_obs_ctx_dec_obs --task carl_classic_cartpole --env.carl.context vary_single --seed 0 --logdir logs/carl_classic_cartpole_vary_single_enc_obs_ctx_dec_obs/0 --wandb.group carl_classic_cartpole_vary_single_enc_obs_ctx_dec_obs --jax.policy_devices 0 --jax.train_devices 1 --run.steps 25000

python -m contextual_mbrl.dreamer.train --configs carl enc_obs_ctx_dec_obs --task carl_classic_cartpole --env.carl.context default --seed 0 --logdir logs/carl_classic_cartpole_default_enc_obs_ctx_dec_obs/0 --wandb.group carl_classic_cartpole_default_enc_obs_ctx_dec_obs --jax.policy_devices 0 --jax.train_devices 1 --run.steps 25000

python -m contextual_mbrl.dreamer.train --configs carl enc_obs_dec_obs --task carl_classic_cartpole --env.carl.context vary_single --seed 1337 --logdir logs/carl_classic_cartpole_vary_single_enc_obs_dec_obs/1337 --wandb.group carl_classic_cartpole_vary_single_enc_obs_dec_obs --jax.policy_devices 0 --jax.train_devices 1 --run.steps 25000

python -m contextual_mbrl.dreamer.train --configs carl enc_obs_dec_obs --task carl_classic_cartpole --env.carl.context vary_single --seed 42 --logdir logs/carl_classic_cartpole_vary_single_enc_obs_dec_obs/42 --wandb.group carl_classic_cartpole_vary_single_enc_obs_dec_obs --jax.policy_devices 0 --jax.train_devices 1 --run.steps 30000

python -m contextual_mbrl.dreamer.train --configs carl enc_obs_dec_obs --task carl_classic_cartpole --env.carl.context default --seed 0 --logdir logs/carl_classic_cartpole_default_enc_obs_dec_obs/0 --wandb.group carl_classic_cartpole_default_enc_obs_dec_obs --jax.policy_devices 0 --jax.train_devices 1 --run.steps 25000


end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime