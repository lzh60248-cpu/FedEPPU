#!/bin/bash
#SBATCH --job-name=main4-s45-agg
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH -e main4_seed45_aggregation_%A_%a.err
#SBATCH -o main4_seed45_aggregation_%A_%a.log

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
  0|1|2) ;;
  *) echo "Submit with: sbatch run-main4-seed45-aggregation-ablation.sh" >&2; exit 2 ;;
esac

VARIANTS=(raw_sample support_aware drop_max_prior)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SEED=45
OUTPUT_DIR="main4_seed45_aggregation_outputs/${SLURM_ARRAY_JOB_ID}/${VARIANT}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main4.py FedEPPU4.py pu_loss.py test_main4_pu.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - "$VARIANT" <<'PY'
import inspect
import sys
import main4
import FedEPPU4

variant = sys.argv[1]
defaults = main4.build_parser().parse_args(["--aggregation_variant", variant])
training_source = inspect.getsource(FedEPPU4.train_fedavg_proposed)
checks = {
    "engine": main4.train_fedavg_proposed is FedEPPU4.train_fedavg_proposed,
    "variant forwarded": "aggregation_variant=args.aggregation_variant" in inspect.getsource(main4.main),
    "ablation function": hasattr(FedEPPU4, "aggregation_ablation_weights"),
    "EMA removed": not hasattr(FedEPPU4, "_update_model_ema"),
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
    "server selects variant": "aggregation_ablation_weights" in training_source,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main4 files on server are out of sync: " + ", ".join(failed))
print(f"preflight=ok; variant={variant}; seed=45; base=Main2")
print("variants=raw_sample,support_aware,drop_max_prior; tasks=3")
print("drop_max_prior=diagnostic_upper_bound_only")
PY

python -m unittest -v test_main4_pu

echo "array_task=$SLURM_ARRAY_TASK_ID; variant=$VARIANT; dataset=cifar10; seed=$SEED"
srun python -u main4.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --aggregation_variant "$VARIANT" \
  --output "$OUTPUT_DIR/main4_${VARIANT}_cifar10_seed45.png"
