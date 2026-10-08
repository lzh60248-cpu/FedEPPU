#!/bin/bash
#SBATCH --job-name=main2-loss
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e main2_local_loss_%A_%a.err
#SBATCH -o main2_local_loss_%A_%a.log

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
  0|1|2|3|4) ;;
  *) echo "Submit with: sbatch run-main2-local-losses.sh" >&2; exit 2 ;;
esac

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="main2_local_loss_outputs/${SLURM_ARRAY_JOB_ID}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main2_loss_curves.py FedEPPU2_loss_curves.py pu_loss.py test_main2_loss_curves.py plot_main2_local_losses.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - <<'PY'
import inspect
import main2_loss_curves
import FedEPPU2_loss_curves

defaults = main2_loss_curves.build_parser().parse_args([])
training_source = inspect.getsource(FedEPPU2_loss_curves.train_fedavg_proposed)
checks = {
    "engine": main2_loss_curves.train_fedavg_proposed is FedEPPU2_loss_curves.train_fedavg_proposed,
    "classification loss": "classification_loss_mean" in training_source,
    "ranking loss": "ranking_loss_mean" in training_source,
    "joint loss": "joint_loss_mean" in training_source,
    "EMA removed": not hasattr(FedEPPU2_loss_curves, "_update_model_ema"),
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main2 loss-curve files out of sync: " + ", ".join(failed))
print("preflight=ok; algorithm=Main2; added_behavior=read-only loss recording")
print("dataset=cifar10; seeds=42,43,44,45,46; rounds=200; tasks=5")
PY

python -m unittest -v test_main2_loss_curves

echo "array_task=$SLURM_ARRAY_TASK_ID; dataset=cifar10; seed=$SEED"
srun python -u main2_loss_curves.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/main2_local_loss_cifar10_seed${SEED}.png"
