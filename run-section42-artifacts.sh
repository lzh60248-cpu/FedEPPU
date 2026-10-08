#!/bin/bash
#SBATCH --job-name=section42-artifacts
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e section42_artifacts_%j.err
#SBATCH -o section42_artifacts_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

INPUT_ROOT="${1:-section42_outputs}"
OUTPUT_DIR="${2:-section42_paper_artifacts}"

for REQUIRED_FILE in build_section42_artifacts.py summarize_section42.py plot_section42_curves.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

if [ ! -d "$INPUT_ROOT" ]; then
  echo "missing_input_root=$INPUT_ROOT" >&2
  exit 2
fi

echo "section=4.2-artifacts"
echo "input_root=$INPUT_ROOT"
echo "output_dir=$OUTPUT_DIR"
echo "execution=single_slurm_job; array=off"

python -u build_section42_artifacts.py \
  --input_root "$INPUT_ROOT" \
  --output_dir "$OUTPUT_DIR"

echo "completed_artifacts=$OUTPUT_DIR"
