#!/bin/bash
#SBATCH --job-name=threshold-seed-plot
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e threshold_seed_plot_%j.err
#SBATCH -o threshold_seed_plot_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""

if [ ! -f plot_threshold_per_seed_acc_auc_f1.py ]; then
  echo "missing_required_file=plot_threshold_per_seed_acc_auc_f1.py" >&2
  exit 2
fi

python -u plot_threshold_per_seed_acc_auc_f1.py \
  --input-root threshold_ablation_outputs \
  --output-dir threshold_ablation_artifacts
