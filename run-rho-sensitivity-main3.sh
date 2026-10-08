#!/bin/bash
#SBATCH --job-name=rho-main3
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%3
#SBATCH -e rho_sensitivity_%A_%a.err
#SBATCH -o rho_sensitivity_%A_%a.log

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
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 19 ]; then
  echo "Submit with: sbatch run-rho-sensitivity-main3.sh" >&2
  exit 2
fi

RHO_NAMES=(0125 0250 0375 0500)
RHO_VALUES=(0.125 0.25 0.375 0.5)
SEEDS=(42 43 44 45 46)
RHO_INDEX=$((TASK_ID / 5))
SEED_INDEX=$((TASK_ID % 5))
RHO_NAME="${RHO_NAMES[$RHO_INDEX]}"
RHO="${RHO_VALUES[$RHO_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="rho_sensitivity_outputs/${SLURM_ARRAY_JOB_ID}/rho${RHO_NAME}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in rho_sensitivity_main.py FedEPPU_rho_sensitivity.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=positive_allocation_ratio_sensitivity; task=$TASK_ID/19"
echo "method=FedEPPU-Main3; dataset=cifar10; rho=$RHO; seed=$SEED"
echo "rounds=200; clients=3; alpha=0.1; batch=256; local_epochs=6; max_steps=15"

srun python -u rho_sensitivity_main.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --positive_ratio "$RHO" \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/fedeppu_cifar10_rho${RHO_NAME}_seed${SEED}.png"

echo "completed rho=$RHO seed=$SEED output=$OUTPUT_DIR"
