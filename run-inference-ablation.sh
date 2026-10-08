#!/bin/bash
#SBATCH --job-name=inference-ablation
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-24%3
#SBATCH -e inference_ablation_%A_%a.err
#SBATCH -o inference_ablation_%A_%a.log

set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/gpfs/home/W125211040/FedEPPU}"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=""
export CIFAR_PATH="$PROJECT_ROOT/data"
export DATASET_PATH="$PROJECT_ROOT/data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

VARIANTS=(full no_ema last_ema zero_threshold last_raw_zero)
SEEDS=(42 43 44 45 46)
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="inference_ablation_outputs/${VARIANT}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in inference_ablation_main.py FedEPPU_inference_ablation.py pu_loss.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "project_root=$PROJECT_ROOT; task=$SLURM_ARRAY_TASK_ID; variant=$VARIANT; seed=$SEED"
srun python -u inference_ablation_main.py \
  --variant "$VARIANT" --dataset cifar10 --rounds 200 --n_clients 3 \
  --seed "$SEED" --alpha 0.1 --batch_size 256 --local_epochs 6 \
  --max_steps_per_epoch 15 \
  --output "$OUTPUT_DIR/${VARIANT}_cifar10_seed${SEED}.png"
