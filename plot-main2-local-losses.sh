#!/bin/bash
#SBATCH --job-name=plot-main2-loss
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e plot_main2_local_loss_%j.err
#SBATCH -o plot_main2_local_loss_%j.log

set -eo pipefail

TRAIN_JOB_ID="${1:?Usage: sbatch plot-main2-local-losses.sh TRAIN_ARRAY_JOB_ID}"
# Prevent the sourced legacy conda activate script from consuming the SLURM
# training job ID as though it were an environment name.
set --

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

ROOT="main2_local_loss_outputs/${TRAIN_JOB_ID}"

python -u plot_main2_local_losses.py \
  --root "$ROOT" \
  --rounds 200 \
  --window 10 \
  --output "$ROOT/figure2_local_losses.pdf"
