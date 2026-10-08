#!/bin/bash
#SBATCH --job-name=main1-steps20-30
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%6
#SBATCH -e cifar10_main1_steps20_30_%A_%a.err
#SBATCH -o cifar10_main1_steps20_30_%A_%a.log

set -eo pipefail
echo "script_version=main1-steps20-30-v1; task=${SLURM_ARRAY_TASK_ID:-unset}"
case "${SLURM_ARRAY_TASK_ID:-}" in
  0|1|2|3|4|5) ;;
  *) echo 'Submit: sbatch --array=0-5%6 run-Fed1.sh' >&2; exit 2 ;;
esac

export PYTHONPATH=""
export CIFAR_PATH=/gpfs/home/W125211040/FedMNPU/data
export DATASET_PATH=/gpfs/home/W125211040/FedMNPU/data
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedMNPU || exit 1

SEEDS=(42 43 44)
STEP_CAPS=(20 30)
# Tasks 0/1/2: cap 20, seeds 42/43/44; tasks 3/4/5: cap 30.
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CAP_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MAX_STEPS="${STEP_CAPS[$CAP_INDEX]}"
DATASET=cifar10
ALPHA=0.1
SEED="${SEEDS[$SEED_INDEX]}"
ROUNDS=200
ANNEALING_HORIZON=200
OUTPUT_DIR="main1_steps20_30_outputs/${SLURM_ARRAY_JOB_ID}/steps${MAX_STEPS}"
mkdir -p "$OUTPUT_DIR"

echo "Python Path: $(which python)"
python -c "import torch; print('Torch OK! Version:', torch.__version__)"
python - <<'PY'
import inspect
import main1
import FedEPPU1
import datasets.dataset as dataset

defaults = main1.build_parser().parse_args([])
checks = {
    "engine": main1.train_fedavg_proposed is FedEPPU1.train_fedavg_proposed,
    "step cap overrides": all(main1.build_parser().parse_args(["--max_steps_per_epoch", str(cap)]).max_steps_per_epoch == cap for cap in (20, 30)),
    "sampler": hasattr(FedEPPU1, "sample_pu_indices"),
    "main1.py dataset option": "--dataset" in main1.build_parser()._option_string_actions,
    "main1.py horizon option": "--annealing_horizon" in main1.build_parser()._option_string_actions,
    "FedEPPU1.py horizon support": "annealing_horizon" in inspect.signature(FedEPPU1.train_fedavg_proposed).parameters,
    "pu_loss.py": FedEPPU1._PU_LOSS_MODULE == "pu_loss",
    "selection interval option": "--selection_interval" in main1.build_parser()._option_string_actions,
    "selection tolerance option": "--selection_auc_tolerance" in main1.build_parser()._option_string_actions,
    "known prior default": defaults.n_shrink == 0.0,
    "unconstrained ACC default": defaults.proxy_min_specificity == 0.0,
    "sample-only aggregation": len(inspect.signature(FedEPPU1.client_weights).parameters) == 1,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main1 files on server are out of sync: " + ", ".join(failed))
print("preflight=ok; program=main1.py; dataset=cifar10; caps=20,30; seeds=42,43,44; runs=6")
PY

echo "array_task=$SLURM_ARRAY_TASK_ID; dataset=$DATASET; alpha=$ALPHA; seed=$SEED; max_steps_per_epoch=$MAX_STEPS; rounds=$ROUNDS; annealing_horizon=$ANNEALING_HORIZON"

python -m unittest -v test_main1_pu

srun python -u main1.py \
  --dataset "$DATASET" \
  --rounds "$ROUNDS" \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha "$ALPHA" \
  --n_shrink 0 \
  --proxy_min_specificity 0 \
  --threshold_objective acc \
  --annealing_horizon "$ANNEALING_HORIZON" \
  --selection_interval 5 \
  --selection_auc_tolerance 0.001 \
  --batch_size 256 \
  --max_steps_per_epoch "$MAX_STEPS" \
  --local_epochs 6 \
  --output "$OUTPUT_DIR/main1_cifar10_alpha0.1_steps${MAX_STEPS}_seed${SEED}.png"
