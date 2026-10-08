#!/bin/bash
#SBATCH --job-name=main4-localq
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e main4_localsupport_%A_%a.err
#SBATCH -o main4_localsupport_%A_%a.log

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
  *) echo "Submit with: sbatch run-main4-localsupport.sh" >&2; exit 2 ;;
esac

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="main4_localsupport_outputs/${SLURM_ARRAY_JOB_ID}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main4.py FedEPPU4.py pu_loss.py test_main4_pu.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - <<'PY'
import inspect
import main4
import FedEPPU4

defaults = main4.build_parser().parse_args([])
objective_source = inspect.getsource(
    FedEPPU4.ClassificationRankingPUObjective.compute
)
training_source = inspect.getsource(FedEPPU4.train_fedavg_proposed)
checks = {
    "engine": main4.train_fedavg_proposed is FedEPPU4.train_fedavg_proposed,
    "local support scale": "new_tensor(support_value)" in objective_source,
    "ranking uses support": "ranking_scale * ranking_loss" in objective_source,
    "server keeps base weights": (
        "round_aggregation_weights = aggregation_weights.copy()" in training_source
    ),
    "prior projection absent": not hasattr(
        FedEPPU4.ComplementMassAggregation, "prior_consistent_weights"
    ),
    "EMA removed": not hasattr(FedEPPU4, "_update_model_ema"),
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main4 files on server are out of sync: " + ", ".join(failed))
print("preflight=ok; program=main4.py; engine=FedEPPU4.py")
print("dataset=cifar10; seeds=42,43,44,45,46; runs=5")
print("local_loss=nnPU+q*ranking; q=min(1,(1-prior)*batch_unlabeled_count)")
print("aggregation=complement_mass; support_aggregation=off; EMA=off")
PY

python -m unittest -v test_main4_pu

echo "array_task=$SLURM_ARRAY_TASK_ID; program=main4.py; dataset=cifar10; seed=$SEED"
srun python -u main4.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/main4_localsupport_cifar10_seed${SEED}.png"
