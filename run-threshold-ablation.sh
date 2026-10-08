#!/bin/bash
#SBATCH --job-name=threshold-ablation
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e threshold_ablation_%A_%a.err
#SBATCH -o threshold_ablation_%A_%a.log

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
case "$TASK_ID" in
  0|1|2|3|4) ;;
  *) echo "Submit with: sbatch run-threshold-ablation.sh" >&2; exit 2 ;;
esac

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$TASK_ID]}"
OUTPUT_DIR="threshold_ablation_outputs/${SLURM_ARRAY_JOB_ID}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in threshold_ablation_main.py FedEPPU_threshold_ablation.py \
  pu_loss.py datasets/dataset.py test_threshold_ablation.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

echo "experiment=4.4.2_threshold; task=$TASK_ID/4; dataset=cifar10; seed=$SEED"
echo "paired_evaluation=calibrated_vs_zero; training_runs=1_per_seed; rounds=200"
srun python -u threshold_ablation_main.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/threshold_cifar10_seed${SEED}.png"

echo "completed seed=$SEED output=$OUTPUT_DIR"
