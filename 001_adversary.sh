#!/bin/bash

#SBATCH --array=0-119
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_vayuu
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

tasks=("carl_dmc_walker" "carl_classic_cartpole")
seeds=("0" "42" "1337" "13" "71" "1994" "1997" "908" "2102" "3")
schemes=("enc_img_dec_img_pgm_ctx_adv" "enc_obs_dec_obs_pgm_ctx_adv")
contexts=("single_0" "single_1" "double_box")

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

group_name="${task}_${context}_${scheme}_normalized"

if [ "$task" == "carl_dmc_walker" ]; then
    steps=100000
else
    steps=50000
fi

python -m contextual_mbrl.dreamer.train --configs carl $scheme --task $task --env.carl.context $context --seed $seed --logdir logs/$group_name/$seed --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 0 --run.steps $steps --wandb.project ""
python -m contextual_mbrl.dreamer.eval --logdir logs/$group_name/$seed
python -m contextual_mbrl.dreamer.record_latents --logdir logs/$group_name/$seed

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime