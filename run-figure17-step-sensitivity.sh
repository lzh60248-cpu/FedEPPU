#!/bin/bash
#SBATCH --job-name=figure17-step
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e figure17_step_%j.err
#SBATCH -o figure17_step_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""

REQUIRED_FILE="build_figure17_step_sensitivity.py"
if [ ! -f "$REQUIRED_FILE" ]; then
  echo "missing_required_file=$REQUIRED_FILE" >&2
  exit 2
fi

python -u build_figure17_step_sensitivity.py \
  --input-root step_trend_main3_outputs \
  --output-root figure17_step_sensitivity_artifacts
