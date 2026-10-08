#!/bin/bash
#SBATCH --job-name=prior-main3
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%3
#SBATCH -e prior_main3_%A_%a.err
#SBATCH -o prior_main3_%A_%a.log

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
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 29 ]; then
  echo "Submit with: sbatch run-prior-heterogeneity-main3.sh" >&2
  exit 2
fi

CONFIGS=(low medium extreme)
METHODS=(fedavg_nnpu fedeppu)
SEEDS=(42 43 44 45 46)
CONFIG_INDEX=$((TASK_ID / 10))
WITHIN_CONFIG=$((TASK_ID % 10))
METHOD_INDEX=$((WITHIN_CONFIG / 5))
SEED_INDEX=$((WITHIN_CONFIG % 5))
CONFIG="${CONFIGS[$CONFIG_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

OUTPUT_DIR="prior_heterogeneity_main3_outputs/${SLURM_ARRAY_JOB_ID}/${CONFIG}/${METHOD}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in prior_heterogeneity_main3.py main3.py FedEPPU3.py pu_loss.py \
  other/baseline_runtime.py other/fedavg_nnpu.py other/fedavg_pu.py other/nnpu_common.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=4.3.1; task=$TASK_ID/29; config=$CONFIG; method=$METHOD; seed=$SEED"
echo "dataset=cifar10; rounds=200; clients=3; batch=256; local_epochs=6; max_steps=15"
srun python -u prior_heterogeneity_main3.py \
  --method "$METHOD" \
  --prior_config "$CONFIG" \
  --seed "$SEED" \
  --rounds 200 \
  --batch_size 256 \
  --local_epochs 6 \
  --max_steps_per_epoch 15 \
  --output "$OUTPUT_DIR/${METHOD}_${CONFIG}_seed${SEED}.pt"

echo "completed config=$CONFIG method=$METHOD seed=$SEED output=$OUTPUT_DIR"
