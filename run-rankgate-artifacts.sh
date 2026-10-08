#!/bin/bash
#SBATCH --job-name=rankgate-sum
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e rankgate_summary_%j.err
#SBATCH -o rankgate_summary_%j.log

set -eo pipefail
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

RUN_ID="$(find rankgate_outputs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -nr | head -n 1)"
if [ -z "$RUN_ID" ]; then
  echo "no_rankgate_output_found; run sbatch run-rankgate-main3.sh first" >&2
  exit 2
fi
srun python -u summarize_rankgate.py --input_dir "rankgate_outputs/$RUN_ID"
