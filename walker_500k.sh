#!/bin/bash

#SBATCH --array=14-239
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_odin
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

tasks=("carl_dmc_walker")
seeds=("0" "42" "1337" "13" "71" "1994" "1997" "908" "2102" "3")
schemes=("enc_obs_dec_obs_default" "enc_img_dec_img_default" "enc_obs_dec_obs" "enc_img_dec_img" "enc_obs_ctx_dec_obs_ctx" "enc_img_ctx_dec_img_ctx" "enc_obs_dec_obs_pgm_ctx" "enc_img_dec_img_pgm_ctx")
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

group_name="${task}_${context}_${scheme}_500k"

if [ "$scheme" == "enc_obs_dec_obs_default" ]; then
    scheme="enc_obs_dec_obs"
    context="default"
elif [ "$scheme" == "enc_img_dec_img_default" ]; then
    scheme="enc_img_dec_img"
    context="default"
fi

# start training only if logdir does not exist
# if training dir doesn't exist copy "${task}_${context}_${scheme}_normalized" to the new logdir

if [ ! -d logs/$group_name/ ]; then
    mkdir -p logs/$group_name
fi

if [ ! -d logs/$group_name/$seed ]; then
    python -m contextual_mbrl.dreamer.train --configs carl $scheme --task $task --env.carl.context $context --seed $seed --logdir logs/$group_name/$seed --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 0 --run.steps 500000 --wandb.project ''
fi

# if eval.jsonl doesn't exist or doesn't have 84 lines
if [ ! -f logs/$group_name/$seed/eval.jsonl ] || [ $(wc -l < logs/$group_name/$seed/eval.jsonl) -ne 84 ]; then
    rm logs/$group_name/$seed/eval.jsonl
    python -m contextual_mbrl.dreamer.eval --logdir logs/$group_name/$seed
fi


end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime