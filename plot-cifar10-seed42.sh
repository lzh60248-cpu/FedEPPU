#!/bin/bash
#SBATCH --job-name=plot-c10-s42
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e plot_cifar10_seed42_%j.err
#SBATCH -o plot_cifar10_seed42_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

if [ ! -f plot_cifar10_seed42.py ]; then
  echo "missing_required_file=plot_cifar10_seed42.py" >&2
  exit 2
fi
if [ ! -d section42_outputs ]; then
  echo "missing_input_directory=section42_outputs" >&2
  exit 2
fi

python -u plot_cifar10_seed42.py \
  --input_dir section42_outputs \
  --output_dir section42_seed42_figure
