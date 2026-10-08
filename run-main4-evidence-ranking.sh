#!/bin/bash
#SBATCH --job-name=main4-evidrank
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH -e main4_evidence_ranking_%A_%a.err
#SBATCH -o main4_evidence_ranking_%A_%a.log

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
  *) echo "Submit with: sbatch run-main4-evidence-ranking.sh" >&2; exit 2 ;;
esac

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="main4_evidence_ranking_outputs/${SLURM_ARRAY_JOB_ID}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in main4.py FedEPPU4.py pu_loss4.py pu_loss.py test_main4_pu.py datasets/dataset.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

python - <<'PY'
import inspect
import main4
import FedEPPU4
import pu_loss4

defaults = main4.build_parser().parse_args([])
ranking_source = inspect.getsource(pu_loss4.pu_pairwise_ranking_loss)
training_source = inspect.getsource(FedEPPU4.train_fedavg_proposed)
checks = {
    "engine": main4.train_fedavg_proposed is FedEPPU4.train_fedavg_proposed,
    "private loss module": FedEPPU4._PU_LOSS_MODULE == "pu_loss4",
    "denominator removed from optimized loss": "raw_evidence_risk.abs()" in ranking_source,
    "reconstructed risk diagnostic only": "reconstructed_ranking_risk" in ranking_source,
    "Main2 aggregation retained": "round_aggregation_weights = aggregation_weights.copy()" in training_source,
    "EMA removed": not hasattr(FedEPPU4, "_update_model_ema"),
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main4 files on server are out of sync: " + ", ".join(failed))
print("preflight=ok; program=main4.py; engine=FedEPPU4.py; loss=pu_loss4.py")
print("base=Main2; single_change=negative-evidence-normalized ranking")
print("optimized_ranking=abs(R_PU-prior*R_PP); tunable_parameter=none")
print("dataset=cifar10; seeds=42,43,44,45,46; rounds=200; tasks=5")
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
  --output "$OUTPUT_DIR/main4_evidence_ranking_cifar10_seed${SEED}.png"
