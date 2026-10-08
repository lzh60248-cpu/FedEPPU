#!/bin/bash
#SBATCH --job-name=prior-heterogeneity
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%3
#SBATCH -e prior_heterogeneity_%A_%a.err
#SBATCH -o prior_heterogeneity_%A_%a.log

set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/gpfs/home/W125211040/FedEPPU}"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""
export CIFAR_PATH="$PROJECT_ROOT/data"
export DATASET_PATH="$PROJECT_ROOT/data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

CONFIGS=(low medium extreme)
SEEDS=(42 43 44 45 46)
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
CONFIG="${CONFIGS[$CONFIG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="prior_heterogeneity_outputs/fedeppu/${CONFIG}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in prior_heterogeneity.py main.py FedEPPU.py pu_loss.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "project_root=$PROJECT_ROOT; task=$SLURM_ARRAY_TASK_ID; method=fedeppu; config=$CONFIG; seed=$SEED"
srun python -u prior_heterogeneity.py \
  --prior_config "$CONFIG" --seed "$SEED" \
  --rounds 200 --batch_size 256 --local_epochs 6 \
  --max_steps_per_epoch 15 \
  --output "$OUTPUT_DIR/fedeppu_${CONFIG}_seed${SEED}.pt"
