#!/bin/bash
#SBATCH --job-name=dirichlet-main3
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%3
#SBATCH -e dirichlet_main3_%A_%a.err
#SBATCH -o dirichlet_main3_%A_%a.log

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
  echo "Submit with: sbatch run-dirichlet-main3.sh" >&2
  exit 2
fi

ALPHAS=(0.1 0.5 1.0)
ALPHA_NAMES=(alpha0p1 alpha0p5 alpha1p0)
SEEDS=(42 43 44 45 46)

DATASET=cifar10
ALPHA_INDEX=$((TASK_ID / 5))
SEED_INDEX=$((TASK_ID % 5))
ALPHA="${ALPHAS[$ALPHA_INDEX]}"
ALPHA_NAME="${ALPHA_NAMES[$ALPHA_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

OUTPUT_DIR="dirichlet_main3_outputs/${SLURM_ARRAY_JOB_ID}/${DATASET}/${ALPHA_NAME}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main3.py FedEPPU3.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "section=dirichlet_heterogeneity; task=$TASK_ID/14"
echo "method=FedEPPU-Main3; dataset=$DATASET; alpha=$ALPHA; seed=$SEED"
echo "rounds=200; clients=3; batch=256; local_epochs=6; max_steps=15"

srun python -u main3.py \
  --dataset "$DATASET" \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha "$ALPHA" \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/fedeppu_${DATASET}_${ALPHA_NAME}_seed${SEED}.png"

echo "completed dataset=$DATASET alpha=$ALPHA seed=$SEED output=$OUTPUT_DIR"
