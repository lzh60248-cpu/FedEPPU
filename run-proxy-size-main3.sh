#!/bin/bash
#SBATCH --job-name=proxy-size
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%3
#SBATCH -e proxy_size_%A_%a.err
#SBATCH -o proxy_size_%A_%a.log

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
  echo "Submit with: sbatch run-proxy-size-main3.sh" >&2
  exit 2
fi

PROXY_SIZES=(128 256 512 1024)
SEEDS=(42 43 44 45 46)
PROXY_POOL_SIZE=1024

SIZE_INDEX=$((TASK_ID / 5))
SEED_INDEX=$((TASK_ID % 5))
N_PROXY="${PROXY_SIZES[$SIZE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

OUTPUT_DIR="proxy_size_main3_outputs/proxy${N_PROXY}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main3.py FedEPPU3.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -c 'import inspect; from datasets.dataset import load_dataset; signature = inspect.signature(load_dataset); assert "proxy_pool_size" in signature.parameters, "datasets/dataset.py is outdated: load_dataset lacks proxy_pool_size"; print("dataset_module=" + str(inspect.getsourcefile(load_dataset))); print("load_dataset_signature=" + str(signature))'

echo "experiment=proxy_size_sensitivity; task=$TASK_ID/19"
echo "method=FedEPPU-Main3; dataset=cifar10; proxy_size=$N_PROXY; seed=$SEED"
echo "reserved_proxy_pool=$PROXY_POOL_SIZE; client_partition=fixed_within_seed"
echo "rounds=200; clients=3; alpha=0.1; batch=256; local_epochs=6; max_steps=15"

srun python -u main3.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --n_proxy "$N_PROXY" \
  --proxy_pool_size "$PROXY_POOL_SIZE" \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/fedeppu_cifar10_proxy${N_PROXY}_seed${SEED}.png"

echo "completed proxy_size=$N_PROXY seed=$SEED output=$OUTPUT_DIR"
