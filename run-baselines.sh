#!/bin/bash
#SBATCH --job-name=all-comparisons
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-44%3
#SBATCH -e comparison_%A_%a.err
#SBATCH -o comparison_%A_%a.log

set -eo pipefail
export PYTHONPATH=""
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
PROJECT_ROOT="/gpfs/home/W125211040/FedEPPU"
cd "$PROJECT_ROOT" || exit 1
export CIFAR_PATH="$PROJECT_ROOT/data"
export DATASET_PATH="$PROJECT_ROOT/data"
echo "project_root=$PROJECT_ROOT"

METHODS=(centralized_pn centralized_nnpu local_nnpu fedavg_nnpu fedprox_nnpu scaffold_nnpu fedpu mrfl_nnpu fedeppu)
SEEDS=(42 43 44 45 46)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
METHOD="${METHODS[$METHOD_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
OUTPUT_DIR="comparison_main2_outputs/cifar10/${METHOD}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

REQUIRED_FILES=(
  other/__init__.py
  other/run_baselines.py
  other/data_protocol.py
  other/centralized.py
  other/local_nnpu.py
  other/fedavg_nnpu.py
  other/fedprox_nnpu.py
  other/scaffold_nnpu.py
  other/fedpu.py
  other/mrfl_nnpu.py
  other/baseline_runtime.py
  other/nnpu_common.py
  other/standard_nnpu.py
  other/fedavg_pu.py
  other/fedprox_pu.py
  other/scaffold_pu.py
  other/fedpu_pu.py
  other/MRFL.py
  main2.py
  FedEPPU2.py
  pu_loss.py
  test_fedpu.py
)
MISSING_FILES=()
for REQUIRED_FILE in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    MISSING_FILES+=("$REQUIRED_FILE")
  fi
done
if [ "${#MISSING_FILES[@]}" -ne 0 ]; then
  echo "Required baseline files are missing on server:" >&2
  printf '  %s\n' "${MISSING_FILES[@]}" >&2
  exit 2
fi

if [ -f other/test_baselines.py ]; then
  python -m unittest -v other.test_baselines test_fedpu
else
  echo "warning=other/test_baselines.py is not synchronized; tests skipped"
fi
python - <<'PY'
import main2
import FedEPPU2
from other.fedpu_pu import FedPUBinaryLoss

assert main2.train_fedavg_proposed is FedEPPU2.train_fedavg_proposed
source = open("FedEPPU2.py", encoding="utf-8").read()
assert "round_aggregation_weights = aggregation_weights.copy()" in source
assert "_update_model_ema" not in source
assert "model_ema_momentum" not in source
assert FedPUBinaryLoss.__name__ == "FedPUBinaryLoss"
print("preflight=ok; fedeppu=main2_no_support_no_ema; fedpu=binary_source_adaptation")
PY

echo "method=${METHOD}; dataset=cifar10; seed=${SEED}"

if [ "$METHOD" = "fedeppu" ]; then
  srun python -u main2.py \
    --dataset cifar10 --rounds 200 --n_clients 3 --seed "$SEED" \
    --alpha 0.1 --batch_size 256 --local_epochs 6 \
    --max_steps_per_epoch 15 \
    --output "$OUTPUT_DIR/fedeppu_cifar10_seed${SEED}.png"
else
  srun python -u -m other.run_baselines \
    --method "$METHOD" --dataset cifar10 --rounds 200 --n_clients 3 \
    --seed "$SEED" --alpha 0.1 --batch_size 256 --local_epochs 6 \
    --max_steps_per_epoch 15 \
    --output "$OUTPUT_DIR/${METHOD}_cifar10_seed${SEED}.pt"
fi
