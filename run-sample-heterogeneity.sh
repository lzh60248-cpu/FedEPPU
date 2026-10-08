#!/bin/bash
#SBATCH --job-name=sample-heterogeneity
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-59%3
#SBATCH -e sample_heterogeneity_%A_%a.err
#SBATCH -o sample_heterogeneity_%A_%a.log

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

SIZE_CONFIGS=(balanced moderate extreme)
AGGREGATIONS=(fedavg complement support joint)
SEEDS=(42 43 44 45 46)
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / 20))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 20))
AGGREGATION_INDEX=$((REMAINDER / 5))
SEED_INDEX=$((REMAINDER % 5))
SIZE_CONFIG="${SIZE_CONFIGS[$SIZE_INDEX]}"
AGGREGATION="${AGGREGATIONS[$AGGREGATION_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="sample_heterogeneity_outputs/${SIZE_CONFIG}/${AGGREGATION}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in sample_heterogeneity.py FedEPPU_ablation.py pu_loss.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "project_root=$PROJECT_ROOT; task=$SLURM_ARRAY_TASK_ID; size=${SIZE_CONFIG}; aggregation=${AGGREGATION}; seed=${SEED}"
srun python -u sample_heterogeneity.py \
  --size_config "$SIZE_CONFIG" --aggregation "$AGGREGATION" --seed "$SEED" \
  --rounds 200 --batch_size 256 --local_epochs 6 \
  --max_steps_per_epoch 15 \
  --output "$OUTPUT_DIR/fedeppu_${SIZE_CONFIG}_${AGGREGATION}_seed${SEED}.pt"
