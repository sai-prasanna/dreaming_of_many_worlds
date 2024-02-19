#!/bin/bash

#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_array
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

schemes=("enc_img_dec_img" "enc_img_ctx_dec_img_ctx" "enc_img_dec_img_ctx")

contexts=("default" "single_0" "single_1" "double_box")

n_tasks=${#tasks[@]}
n_seeds=${#seeds[@]}
n_schemes=${#schemes[@]}
n_contexts=${#contexts[@]}

for idx in {0..59}
do
    task_index=$((${idx} / (n_seeds * n_schemes * n_contexts) % n_tasks))
    seed_index=$((${idx} / (n_schemes * n_contexts) % n_seeds))
    scheme_index=$((${idx} / n_contexts % n_schemes))
    context_index=$((${idx} % n_contexts))
    task=${tasks[$task_index]}
    seed=${seeds[$seed_index]}
    scheme=${schemes[$scheme_index]}
    context=${contexts[$context_index]}
    group_name="${task}_${context}_${scheme}_normalized"

    if [ -d "logs/$group_name/$seed" ]; then
        python -m contextual_mbrl.dreamer.record_cart_length_dreams --logdir logs/$group_name/$seed
    fi
done


end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime