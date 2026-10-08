#!/bin/bash
#SBATCH --job-name=plot-main-main2
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e plot_main_main2_%j.err
#SBATCH -o plot_main_main2_%j.log

set -eo pipefail

TRAIN_JOB_ID="${1:?Usage: sbatch plot-main-main2-curves.sh TRAIN_ARRAY_JOB_ID}"
# The legacy conda activate script reads the shell's positional parameters.
# Clear them after preserving the training job ID so it cannot treat that ID
# as an environment name.
set --

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

ROOT="main_main2_round_outputs/${TRAIN_JOB_ID}"

python -u plot_main_main2_hpc.py \
  --root "$ROOT" \
  --rounds 200 \
  --window 5 \
  --output "$ROOT/main_vs_main2_five_seed_round_curves.pdf"
