#!/bin/bash

#SBATCH --array=0-24
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_agni
#SBATCH --output logs/slurm/%x-%A-%a-HelloCluster.out
#SBATCH --error logs/slurm/%x-%A-%a-HelloCluster.err
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate c_mbrl

start=`date +%s`

tasks=("carl_box2d_lunar_lander")
seeds=("0" "42" "1337" "13" "71")
schemes=("enc_obs_dec_obs_default" "enc_obs_dec_obs" "enc_obs_ctx_dec_obs_ctx" "enc_obs_dec_obs_pgm_ctx" "enc_obs_dec_obs_pgm_ctx_adv")
contexts=("double_box")

n_tasks=${#tasks[@]}
n_seeds=${#seeds[@]}
n_schemes=${#schemes[@]}
n_contexts=${#contexts[@]}

task_index=$((${SLURM_ARRAY_TASK_ID} / (n_seeds * n_schemes * n_contexts) % n_tasks))
seed_index=$((${SLURM_ARRAY_TASK_ID} / (n_schemes * n_contexts) % n_seeds))
scheme_index=$((${SLURM_ARRAY_TASK_ID} / n_contexts % n_schemes))
context_index=$((${SLURM_ARRAY_TASK_ID} % n_contexts))

task=${tasks[$task_index]}
seed=${seeds[$seed_index]}
scheme=${schemes[$scheme_index]}
context=${contexts[$context_index]}

if [ "$scheme" == "enc_obs_dec_obs_default" ]; then
    scheme="enc_obs_dec_obs"
    context="default"
fi


group_name="${task}_${context}_${scheme}_normalized_250k"


python -m contextual_mbrl.dreamer.train --configs carl $scheme --task $task --env.carl.context $context --seed $seed --logdir logs/$group_name/$seed --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 0 --run.steps 250000 --wandb.project ""
python -m contextual_mbrl.dreamer.eval --logdir logs/$group_name/$seed
python -m contextual_mbrl.dreamer.record_latents --logdir logs/$group_name/$seed

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime