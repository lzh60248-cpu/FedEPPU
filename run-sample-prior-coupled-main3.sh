#!/bin/bash
#SBATCH --job-name=prior-size
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3
#SBATCH -e prior_size_%A_%a.err
#SBATCH -o prior_size_%A_%a.log

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
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 8 ]; then
  echo "Submit with: sbatch --array=0-8%3 run-sample-prior-coupled-main3.sh" >&2
  exit 2
fi

CONFIGS=(balanced moderate extreme)
METHODS=(fedavg_nnpu fedeppu_no_complement fedeppu)
CONFIG_INDEX=$((TASK_ID / 3))
METHOD_INDEX=$((TASK_ID % 3))
CONFIG="${CONFIGS[$CONFIG_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
SEED=44
OUTPUT_DIR="sample_prior_coupled_main3_outputs/${SLURM_ARRAY_JOB_ID}/${CONFIG}/${METHOD}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in sample_prior_coupled_main3.py FedEPPU_sample_main3.py pu_loss.py other/baseline_runtime.py other/fedavg_nnpu.py other/nnpu_common.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=4.3.2-prior-coupled; task=$TASK_ID/8; config=$CONFIG; method=$METHOD; seed=$SEED"
echo "dataset=cifar10; rounds=200; clients=3; batch=256; local_epochs=6; max_steps=15"
srun python -u sample_prior_coupled_main3.py \
  --size_config "$CONFIG" \
  --method "$METHOD" \
  --seed "$SEED" \
  --rounds 200 \
  --batch_size 256 \
  --local_epochs 6 \
  --max_steps_per_epoch 15 \
  --output "$OUTPUT_DIR/${METHOD}_${CONFIG}_seed${SEED}.pt"

echo "completed config=$CONFIG method=$METHOD seed=$SEED output=$OUTPUT_DIR"