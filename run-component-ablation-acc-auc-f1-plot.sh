#!/bin/bash
#SBATCH --job-name=abl-curve
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e component_curve_%j.err
#SBATCH -o component_curve_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""

for REQUIRED_FILE in plot_component_ablation_acc_auc_f1.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -u plot_component_ablation_acc_auc_f1.py \
  --input-root component_ablation_outputs \
  --output-dir component_ablation_artifacts
