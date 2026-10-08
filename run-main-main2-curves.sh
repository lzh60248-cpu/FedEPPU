#!/bin/bash
#SBATCH --job-name=main-main2-curve
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%3
#SBATCH -e main_main2_curve_%A_%a.err
#SBATCH -o main_main2_curve_%A_%a.log

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

case "${SLURM_ARRAY_TASK_ID:-}" in
  0|1|2|3|4|5|6|7|8|9) ;;
  *) echo "Submit with: sbatch run-main-main2-curves.sh" >&2; exit 2 ;;
esac

METHODS=(main main main main main main2 main2 main2 main2 main2)
SEEDS=(42 43 44 45 46 42 43 44 45 46)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

if [ "$METHOD" = "main" ]; then
  PROGRAM="main.py"
  ENGINE="FedEPPU.py"
  TEST_MODULE="test_main_pu"
else
  PROGRAM="main2.py"
  ENGINE="FedEPPU2.py"
  TEST_MODULE="test_main2_pu"
fi

OUTPUT_DIR="main_main2_round_outputs/${SLURM_ARRAY_JOB_ID}/${METHOD}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in "$PROGRAM" "$ENGINE" pu_loss.py "${TEST_MODULE}.py" datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - "$METHOD" <<'PY'
import importlib
import sys

method = sys.argv[1]
main_module = importlib.import_module(method)
engine_module = importlib.import_module("FedEPPU" if method == "main" else "FedEPPU2")
defaults = main_module.build_parser().parse_args([])
checks = {
    "engine connection": main_module.train_fedavg_proposed is engine_module.train_fedavg_proposed,
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
    "round default": defaults.rounds == 200,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"{method} files on server are out of sync: " + ", ".join(failed))
print(f"preflight=ok; method={method}; program={main_module.__file__}; engine={engine_module.__file__}")
PY

python -m unittest -v "$TEST_MODULE"

echo "array_task=$SLURM_ARRAY_TASK_ID; method=$METHOD; dataset=cifar10; seed=$SEED; rounds=200"
srun python -u "$PROGRAM" \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/${METHOD}_cifar10_seed${SEED}.png"
