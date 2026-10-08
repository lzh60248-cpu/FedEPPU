#!/bin/bash
#SBATCH --job-name=section42-plot
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e section42_plot_%j.err
#SBATCH -o section42_plot_%j.log

set -eo pipefail

TRAIN_JOB_ID="${1:-}"
if ! [[ "$TRAIN_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch --dependency=afterok:<TRAIN_JOB_ID> plot-section42.sh <TRAIN_JOB_ID>" >&2
  exit 2
fi
set --

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
INPUT_DIR="section42_outputs/$TRAIN_JOB_ID"
OUTPUT_DIR="$INPUT_DIR/paper_artifacts"

if [ ! -d "$INPUT_DIR" ]; then
  echo "missing_input_directory=$INPUT_DIR" >&2
  exit 2
fi

python -u summarize_section42.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
python -u plot_section42_curves.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

echo "section42_artifacts=$OUTPUT_DIR"
