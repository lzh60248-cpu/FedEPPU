#!/bin/bash
#SBATCH --job-name=priorbalanced
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e priorbalanced_%A_%a.err
#SBATCH -o priorbalanced_%A_%a.log

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

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="priorbalanced_outputs/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main_priorbalanced.py FedEPPU_priorbalanced.py pu_loss.py test_priorbalanced.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python -m unittest -v test_priorbalanced

python - <<'PY'
import main_priorbalanced
import FedEPPU_priorbalanced

assert main_priorbalanced.train_fedavg_proposed is FedEPPU_priorbalanced.train_fedavg_proposed
source = open("FedEPPU_priorbalanced.py", encoding="utf-8").read()
assert "prior_balanced_weights" in source
assert "round_aggregation_weights = aggregation_weights.copy()" in source
assert "_update_model_ema" not in source
assert "model_ema_momentum" not in source
print("preflight=ok; prior_balance=on; negative_support_aggregation=off; model_ema=off")
PY

echo "program=main_priorbalanced.py; dataset=cifar10; seed=${SEED}; max_steps_per_epoch=15"
srun python -u main_priorbalanced.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/priorbalanced_cifar10_seed${SEED}.png"
