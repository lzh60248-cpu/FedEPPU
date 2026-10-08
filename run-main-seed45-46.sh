#!/bin/bash
#SBATCH --job-name=main-s45-46
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH -e main_seed45_46_%A_%a.err
#SBATCH -o main_seed45_46_%A_%a.log

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
  0|1) ;;
  *) echo "Submit with: sbatch run-main-seed45-46.sh" >&2; exit 2 ;;
esac

SEEDS=(45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="main_seed45_46_outputs/${SLURM_ARRAY_JOB_ID}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main.py FedEPPU.py pu_loss.py test_main_pu.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - <<'PY'
import main
import FedEPPU

defaults = main.build_parser().parse_args([])
checks = {
    "engine": main.train_fedavg_proposed is FedEPPU.train_fedavg_proposed,
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
    "support aggregation": hasattr(FedEPPU.ComplementMassAggregation, "support_aware_weights"),
    "EMA": hasattr(FedEPPU, "_update_model_ema"),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main files on server are out of sync: " + ", ".join(failed))
print("preflight=ok; program=main.py; engine=FedEPPU.py")
print("dataset=cifar10; seeds=45,46; max_steps_per_epoch=15; local_epochs=6")
print("support_aggregation=on; EMA=on")
PY

python -m unittest -v test_main_pu

echo "array_task=$SLURM_ARRAY_TASK_ID; program=main.py; dataset=cifar10; seed=$SEED"
srun python -u main.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/main_cifar10_seed${SEED}.png"
