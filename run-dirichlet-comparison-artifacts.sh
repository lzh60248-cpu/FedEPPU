#!/bin/bash
#SBATCH --job-name=dirichlet-figure
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e dirichlet_figure_%j.err
#SBATCH -o dirichlet_figure_%j.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

for REQUIRED_FILE in build_dirichlet_comparison_artifacts.py \
  summarize_plot_dirichlet_comparison_main3.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -u build_dirichlet_comparison_artifacts.py \
  --input_root dirichlet_comparison_outputs \
  --output_dir dirichlet_comparison_artifacts

echo "completed_artifacts=dirichlet_comparison_artifacts"
