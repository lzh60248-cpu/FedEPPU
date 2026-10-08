#!/bin/bash
#SBATCH --job-name=figure1-cifar-seeds
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e figure1_cifar_seeds_%j.err
#SBATCH -o figure1_cifar_seeds_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""

for REQUIRED_FILE in plot_figure1_cifar10_fedeppu_seeds.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -u plot_figure1_cifar10_fedeppu_seeds.py \
  --input-root section42_paper_artifacts/merged_results \
  --output-dir figure1_cifar10_fedeppu_artifacts
