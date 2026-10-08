#!/bin/bash
#SBATCH --job-name=rankdiag
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e rankdiag_%A_%a.err
#SBATCH -o rankdiag_%A_%a.log

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
if ! [[ "$TASK_ID" =~ ^[0-4]$ ]]; then
  echo "Submit with: sbatch run-rankdiag-main3.sh" >&2
  exit 2
fi

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$TASK_ID]}"
OUTPUT_DIR="rankdiag_outputs/${SLURM_ARRAY_JOB_ID}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in rankdiag_main.py FedEPPU_rankdiag.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=extreme_prior_ranking_diagnostics; task=$TASK_ID/4"
echo "method=FedEPPU-Main3-unchanged-training-rule; dataset=cifar10; seed=$SEED"
echo "rounds=200; clients=3; alpha=0.1; batch=256; P=64; U=192; local_epochs=6; max_steps=15"

srun python -u rankdiag_main.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/rankdiag_cifar10_seed${SEED}.png"

echo "completed seed=$SEED output=$OUTPUT_DIR"
