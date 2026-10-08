#!/bin/bash
#SBATCH --job-name=rho-summary
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e rho_summary_%j.err
#SBATCH -o rho_summary_%j.log

set -eo pipefail
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

INPUT_DIR="$(find rho_sensitivity_outputs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -nr | head -n 1)"
if [ -z "$INPUT_DIR" ]; then
  echo "no_rho_sensitivity_output_found" >&2
  exit 2
fi
INPUT_DIR="rho_sensitivity_outputs/$INPUT_DIR"
echo "input_dir=$INPUT_DIR"
srun python -u summarize_rho_sensitivity_main3.py --input_dir "$INPUT_DIR"
