#!/bin/bash
#SBATCH --job-name=main3-loss-plot
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e main3_loss_artifacts_%j.err
#SBATCH -o main3_loss_artifacts_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""

for REQUIRED_FILE in build_main3_local_loss_artifacts.py plot_main3_local_losses.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -u build_main3_local_loss_artifacts.py \
  --input-root main3_local_loss_outputs \
  --output-root main3_local_loss_artifacts
