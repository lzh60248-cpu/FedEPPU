#!/bin/bash
#SBATCH --job-name=rankgate
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%3
#SBATCH -e rankgate_%A_%a.err
#SBATCH -o rankgate_%A_%a.log

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
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 14 ]; then
  echo "Submit with: sbatch run-rankgate-main3.sh" >&2
  exit 2
fi

VARIANTS=(original continuous hard)
SEEDS=(42 43 44 45 46)
VARIANT_INDEX=$((TASK_ID / 5))
SEED_INDEX=$((TASK_ID % 5))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="rankgate_outputs/${SLURM_ARRAY_JOB_ID}/${VARIANT}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in rankgate_main.py FedEPPU_rankgate.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=ranking_gate_comparison; task=$TASK_ID/14"
echo "variant=$VARIANT; seed=$SEED; dataset=cifar10"
echo "rounds=200; clients=3; alpha=0.1; batch=256; P=64; U=192; local_epochs=6; max_steps=15"

srun python -u rankgate_main.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --ranking_gate "$VARIANT" \
  --output "$OUTPUT_DIR/rankgate_${VARIANT}_cifar10_seed${SEED}.png"

echo "completed variant=$VARIANT seed=$SEED output=$OUTPUT_DIR"
