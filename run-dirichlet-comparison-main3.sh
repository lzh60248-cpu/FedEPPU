#!/bin/bash
#SBATCH --job-name=dirichlet-compare
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%3
#SBATCH -e dirichlet_compare_%A_%a.err
#SBATCH -o dirichlet_compare_%A_%a.log

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
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 39 ]; then
  echo "Submit with: sbatch run-dirichlet-comparison-main3.sh" >&2
  exit 2
fi

ALPHAS=(0.1 0.5 1.0 2.0)
ALPHA_NAMES=(alpha0p1 alpha0p5 alpha1p0 alpha2p0)
METHODS=(fedavg_nnpu fedeppu)
SEEDS=(42 43 44 45 46)

ALPHA_INDEX=$((TASK_ID / 10))
WITHIN_ALPHA=$((TASK_ID % 10))
METHOD_INDEX=$((WITHIN_ALPHA / 5))
SEED_INDEX=$((WITHIN_ALPHA % 5))
ALPHA="${ALPHAS[$ALPHA_INDEX]}"
ALPHA_NAME="${ALPHA_NAMES[$ALPHA_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

OUTPUT_DIR="dirichlet_comparison_outputs/${SLURM_ARRAY_JOB_ID}/${ALPHA_NAME}/${METHOD}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main3.py FedEPPU3.py pu_loss.py datasets/dataset.py \
  dirichlet_fedavg_nnpu.py other/fedavg_nnpu.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=dirichlet_prior_heterogeneity; task=$TASK_ID/39"
echo "dataset=cifar10; method=$METHOD; alpha=$ALPHA; seed=$SEED"
echo "rounds=200; clients=3; batch=256; local_epochs=6; max_steps=15"

if [ "$METHOD" = "fedavg_nnpu" ]; then
  srun python -u dirichlet_fedavg_nnpu.py \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --rounds 200 \
    --n_clients 3 \
    --batch_size 256 \
    --max_steps_per_epoch 15 \
    --local_epochs 6 \
    --output "$OUTPUT_DIR/fedavg_nnpu_cifar10_${ALPHA_NAME}_seed${SEED}.pt"
else
  srun python -u main3.py \
    --dataset cifar10 \
    --rounds 200 \
    --n_clients 3 \
    --seed "$SEED" \
    --alpha "$ALPHA" \
    --batch_size 256 \
    --max_steps_per_epoch 15 \
    --local_epochs 6 \
    --output "$OUTPUT_DIR/fedeppu_cifar10_${ALPHA_NAME}_seed${SEED}.png"
fi

echo "completed alpha=$ALPHA method=$METHOD seed=$SEED output=$OUTPUT_DIR"
