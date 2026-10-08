#!/bin/bash
#SBATCH --job-name=main4-steps
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%2
#SBATCH -e main4_step_protocol_%A_%a.err
#SBATCH -o main4_step_protocol_%A_%a.log

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
  *) echo "Submit with: sbatch run-main4-step-protocol.sh" >&2; exit 2 ;;
esac

SEEDS=(42 43 44 45 46)
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
  PROTOCOL="capped_by_samples"
  SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
else
  PROTOCOL="fixed_15"
  SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 5))]}"
fi

OUTPUT_DIR="main4_step_protocol_outputs/${SLURM_ARRAY_JOB_ID}/${PROTOCOL}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main4.py FedEPPU4.py pu_loss.py test_main4_pu.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - "$PROTOCOL" <<'PY'
import inspect
import sys
import main4
import FedEPPU4

protocol = sys.argv[1]
defaults = main4.build_parser().parse_args(["--local_step_protocol", protocol])
main_source = inspect.getsource(main4.main)
engine_source = inspect.getsource(FedEPPU4.train_fedavg_proposed)
checks = {
    "engine": main4.train_fedavg_proposed is FedEPPU4.train_fedavg_proposed,
    "protocol forwarded": "local_step_protocol=args.local_step_protocol" in main_source,
    "protocol used": "local_step_protocol=local_step_protocol" in engine_source,
    "support aggregation": "support_aware_weights" in engine_source,
    "EMA removed": not hasattr(FedEPPU4, "_update_model_ema"),
    "step count": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main4 files on server are out of sync: " + ", ".join(failed))
print(f"preflight=ok; program=main4.py; engine=FedEPPU4.py; protocol={protocol}")
print("base=Main3; support_aggregation=on; EMA=off; dataset=cifar10")
print("seeds=42,43,44,45,46; rounds=200; max_steps_per_epoch=15; tasks=10")
PY

python -m unittest -v test_main4_pu

echo "array_task=$SLURM_ARRAY_TASK_ID; protocol=$PROTOCOL; dataset=cifar10; seed=$SEED"
srun python -u main4.py \
  --dataset cifar10 \
  --rounds 200 \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch 15 \
  --local_epochs 6 \
  --local_step_protocol "$PROTOCOL" \
  --output "$OUTPUT_DIR/main4_${PROTOCOL}_cifar10_seed${SEED}.png"
