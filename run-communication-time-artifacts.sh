#!/bin/bash
#SBATCH --job-name=comm-artifact
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e communication_artifacts_%j.err
#SBATCH -o communication_artifacts_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""

for REQUIRED_FILE in build_communication_time_artifacts.py summarize_communication_time.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -u build_communication_time_artifacts.py \
  --input-root communication_time_outputs \
  --output-root communication_time_artifacts
