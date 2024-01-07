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
group_name="carl_classic_cartpole_default_enc_img_ctx_dec_img_ctx"
seed=0
python -m contextual_mbrl.dreamer.eval --logdir logs/$group_name/$seed --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 1
python -m contextual_mbrl.dreamer.train --configs carl enc_img_ctx_dec_img_ctx --task carl_classic_cartpole --env.carl.context default --seed 10 --logdir logs/$group_name/$seed --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 1 --run.steps 50000
end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime