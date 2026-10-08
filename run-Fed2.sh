#!/bin/bash
#SBATCH --job-name=main2-noema
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e main2_noema_%A_%a.err
#SBATCH -o main2_noema_%A_%a.log

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
OUTPUT_DIR="main2_noema_outputs/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main2.py FedEPPU2.py pu_loss.py test_main2_pu.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - <<'PY'
import main2
import FedEPPU2

assert main2.train_fedavg_proposed is FedEPPU2.train_fedavg_proposed
source = open("FedEPPU2.py", encoding="utf-8").read()
assert "round_aggregation_weights = aggregation_weights.copy()" in source
assert "_update_model_ema" not in source
assert "model_ema_momentum" not in source
assert "eval_model.load_state_dict(copy.deepcopy(global_model.state_dict()))" in source
print("preflight=ok; program=main2.py; negative_support_aggregation=off; model_ema=off")
PY

python -m unittest -v test_main2_pu

echo "program=main2.py; dataset=cifar10; seed=${SEED}; max_steps_per_epoch=15; negative_support_aggregation=off; model_ema=off"
srun python -u main2.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/main2_noema_cifar10_seed${SEED}.png"
