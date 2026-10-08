#!/bin/bash
#SBATCH --job-name=client-count
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%3
#SBATCH -e client_count_%A_%a.err
#SBATCH -o client_count_%A_%a.log

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
  echo "Submit with: sbatch run-client-count-main3.sh" >&2
  exit 2
fi

CLIENT_COUNTS=(3 6 9 12)
SEEDS=(42 43 44 45 46)

CLIENT_INDEX=$((TASK_ID / 5))
SEED_INDEX=$((TASK_ID % 5))
N_CLIENTS="${CLIENT_COUNTS[$CLIENT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

OUTPUT_DIR="client_count_main3_outputs/clients${N_CLIENTS}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main3.py FedEPPU3.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=client_count_ablation; task=$TASK_ID/19"
echo "method=FedEPPU-Main3; dataset=cifar10; clients=$N_CLIENTS; seed=$SEED"
echo "rounds=200; alpha=0.1; batch=256; local_epochs=6; max_steps=15"
echo "participation=all_clients; total_dataset_size=fixed"

srun python -u main3.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients "$N_CLIENTS" \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/fedeppu_cifar10_clients${N_CLIENTS}_seed${SEED}.png"

echo "completed clients=$N_CLIENTS seed=$SEED output=$OUTPUT_DIR"
