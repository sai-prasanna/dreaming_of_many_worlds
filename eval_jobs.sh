#!/bin/bash

#SBATCH --array=0-44
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name CMbRL_eval_fin
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

folders=(
    "logs/carl_dmc_walker_single_1_enc_obs_dec_obs_normalized/2102"
    "logs/carl_dmc_walker_single_1_enc_obs_dec_obs_normalized/3"
    "logs/carl_dmc_walker_single_0_enc_img_dec_img_normalized/908"
    "logs/carl_dmc_walker_single_0_enc_img_dec_img_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_img_dec_img_normalized/3"
    "logs/carl_dmc_walker_single_1_enc_img_dec_img_normalized/908"
    "logs/carl_dmc_walker_single_1_enc_img_dec_img_normalized/2102"
    "logs/carl_dmc_walker_single_1_enc_img_dec_img_normalized/3"
    "logs/carl_dmc_walker_single_1_enc_img_dec_img_pgm_ctx_normalized/908"
    "logs/carl_dmc_walker_single_1_enc_img_dec_img_pgm_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_1_enc_img_dec_img_pgm_ctx_normalized/3"
    "logs/carl_dmc_walker_single_1_enc_obs_dec_obs_pgm_ctx_normalized/908"
    "logs/carl_dmc_walker_single_1_enc_obs_dec_obs_pgm_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_img_dec_img_pgm_ctx_normalized/13"
    "logs/carl_dmc_walker_single_0_enc_img_dec_img_pgm_ctx_normalized/2102"
    "logs/carl_dmc_walker_double_box_enc_obs_dec_obs_pgm_ctx_normalized/1337"
    "logs/carl_dmc_walker_double_box_enc_obs_dec_obs_pgm_ctx_normalized/908"
    "logs/carl_dmc_walker_double_box_enc_obs_dec_obs_pgm_ctx_normalized/2102"
    "logs/carl_dmc_walker_double_box_enc_obs_dec_obs_normalized/2102"
    "logs/carl_dmc_walker_double_box_enc_obs_dec_obs_normalized/3"
    "logs/carl_dmc_walker_double_box_enc_img_ctx_dec_img_ctx_normalized/908"
    "logs/carl_dmc_walker_double_box_enc_img_ctx_dec_img_ctx_normalized/2102"
    "logs/carl_dmc_walker_double_box_enc_obs_ctx_dec_obs_ctx_normalized/2102"
    "logs/carl_dmc_walker_double_box_enc_obs_ctx_dec_obs_ctx_normalized/3"
    "logs/carl_dmc_walker_single_0_enc_obs_dec_obs_pgm_ctx_normalized/908"
    "logs/carl_dmc_walker_single_0_enc_obs_dec_obs_pgm_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_obs_dec_obs_pgm_ctx_normalized/3"
    "logs/carl_dmc_walker_double_box_enc_img_dec_img_normalized/908"
    "logs/carl_dmc_walker_double_box_enc_img_dec_img_normalized/2102"
    "logs/carl_dmc_walker_double_box_enc_img_dec_img_normalized/3"
    "logs/carl_dmc_walker_double_box_enc_img_dec_img_pgm_ctx_normalized/908"
    "logs/carl_dmc_walker_double_box_enc_img_dec_img_pgm_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_obs_dec_obs_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_obs_dec_obs_normalized/3"
    "logs/carl_dmc_walker_single_0_enc_obs_ctx_dec_obs_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_obs_ctx_dec_obs_ctx_normalized/3"
    "logs/carl_dmc_walker_single_1_enc_obs_ctx_dec_obs_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_1_enc_obs_ctx_dec_obs_ctx_normalized/3"
    "logs/carl_dmc_walker_single_0_enc_img_ctx_dec_img_ctx_normalized/42"
    "logs/carl_dmc_walker_single_0_enc_img_ctx_dec_img_ctx_normalized/908"
    "logs/carl_dmc_walker_single_0_enc_img_ctx_dec_img_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_0_enc_img_ctx_dec_img_ctx_normalized/3"
    "logs/carl_dmc_walker_single_1_enc_img_ctx_dec_img_ctx_normalized/908"
    "logs/carl_dmc_walker_single_1_enc_img_ctx_dec_img_ctx_normalized/2102"
    "logs/carl_dmc_walker_single_1_enc_img_ctx_dec_img_ctx_normalized/3"
)

folder = ${folders[${SLURM_ARRAY_TASK_ID}]}

python -m contextual_mbrl.dreamer.eval --logdir $folder --jax.policy_devices 0 --jax.train_devices 0

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime