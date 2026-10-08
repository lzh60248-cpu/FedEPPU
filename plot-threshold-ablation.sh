#!/bin/bash
#SBATCH --job-name=threshold-plot
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e threshold_plot_%j.err
#SBATCH -o threshold_plot_%j.log

set -eo pipefail

TRAIN_JOB_ID="${1:-}"
if ! [[ "$TRAIN_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch plot-threshold-ablation.sh <TRAIN_JOB_ID>" >&2
  exit 2
fi
set --

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
python -u summarize_plot_threshold_ablation.py \
  --input_dir "threshold_ablation_outputs/$TRAIN_JOB_ID" \
  --output_dir "threshold_ablation_outputs/$TRAIN_JOB_ID/paper_artifacts"
