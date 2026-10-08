#!/bin/bash
#SBATCH --job-name=sum-proxy-size
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e summarize_proxy_size_%j.err
#SBATCH -o summarize_proxy_size_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1

INPUT_DIR="proxy_size_main3_outputs"
if [ ! -d "$INPUT_DIR" ]; then
  echo "Missing experiment directory: $INPUT_DIR" >&2
  exit 2
fi
if [ ! -f "summarize_plot_proxy_size_main3.py" ]; then
  echo "Missing summarizer: summarize_plot_proxy_size_main3.py" >&2
  exit 2
fi

python -u summarize_plot_proxy_size_main3.py \
  --input_dir "$INPUT_DIR"

echo "Proxy-size artifacts: $INPUT_DIR/summary"
