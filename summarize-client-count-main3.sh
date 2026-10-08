#!/bin/bash
#SBATCH --job-name=sum-client-count
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e summarize_client_count_%j.err
#SBATCH -o summarize_client_count_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

INPUT_DIR="client_count_main3_outputs"
if [ ! -d "$INPUT_DIR" ]; then
  echo "Missing experiment directory: $INPUT_DIR" >&2
  exit 2
fi
if [ ! -f "summarize_plot_client_count_main3.py" ]; then
  echo "Missing summarizer: summarize_plot_client_count_main3.py" >&2
  exit 2
fi

python -u summarize_plot_client_count_main3.py \
  --input_dir "$INPUT_DIR"

echo "Client-count artifacts: $INPUT_DIR/summary"
