#!/bin/bash

#SBATCH --array=0-9
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_specific_cartpole
#SBATCH --output logs/slurm/%x-%A-%a-HelloCluster.out
#SBATCH --error logs/slurm/%x-%A-%a-HelloCluster.err
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2

echo "Workingdir: $PWD";
echo "Started at $(date)";

source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate c_mbrl

start=`date +%s`

tasks=("carl_classic_cartpole")
seeds=("0" "42" "1337" "13" "71")
schemes=("enc_obs_dec_obs")
# gravity [0.98, 2.45, 3.92, 4.9, 9.8, 14.7, 15.68, 17.64, 19.6]
# length [0.1 0.2 0.3 0.5 0.7 0.8 0.9 1.0]
# grav, len [(2.45, 0.2), (17.64, 0.2), (17.64, 0.9), (2.45, 0.9)]
# contexts=("specific_0-0.98" "specific_0-2.45" "specific_0-3.92" "specific_0-4.9" "specific_0-9.8" "specific_0-14.7" "specific_0-15.68" "specific_0-17.64" "specific_0-19.6" "specific_1-0.1" "specific_1-0.2" "specific_1-0.3" "specific_1-0.5" "specific_1-0.7" "specific_1-0.8" "specific_1-0.9" "specific_1-1.0")

#contexts=("specific_0-2.45_1-0.2" "specific_0-17.64_1-0.2" "specific_0-17.64_1-0.9" "specific_0-2.45_1-0.9")

contexts=("specific_0-2.45_1-0.2" "specific_0-17.64_1-0.9")

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

group_name="${task}_${context}_${scheme}"



python -m contextual_mbrl.dreamer.train --configs carl $scheme --task $task --env.carl.context $context --seed $seed --logdir logs/specific/$group_name/$seed --wandb.project '' --jax.policy_devices 0 --jax.train_devices 1 --run.steps 50000
python -m contextual_mbrl.dreamer.eval --logdir logs/specific/$group_name/$seed

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime