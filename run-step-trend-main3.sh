#!/bin/bash
#SBATCH --job-name=step-trend
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-24%3
#SBATCH -e step_trend_%A_%a.err
#SBATCH -o step_trend_%A_%a.log

set -eo pipefail

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""
export CIFAR_PATH="$PROJECT_ROOT/data"
export DATASET_PATH="$PROJECT_ROOT/data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 24 ]; then
  echo "Submit with: sbatch run-step-trend-main3.sh" >&2
  exit 2
fi

STEP_CAPS=(5 10 15 20 30)
SEEDS=(42 43 44 45 46)
STEP_INDEX=$((TASK_ID / 5))
SEED_INDEX=$((TASK_ID % 5))
STEP_CAP="${STEP_CAPS[$STEP_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="step_trend_main3_outputs/${SLURM_ARRAY_JOB_ID}/steps${STEP_CAP}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main3.py FedEPPU3.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=local_step_trend; task=$TASK_ID/24"
echo "method=FedEPPU-Main3; dataset=cifar10; step_cap=$STEP_CAP; seed=$SEED"
echo "rounds=200; clients=3; alpha=0.1; batch=256; local_epochs=6"

srun python -u main3.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch "$STEP_CAP" \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/fedeppu_cifar10_steps${STEP_CAP}_seed${SEED}.png"

echo "completed step_cap=$STEP_CAP seed=$SEED output=$OUTPUT_DIR"
