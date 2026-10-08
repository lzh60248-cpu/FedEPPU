#!/bin/bash
#SBATCH --job-name=main9-fourdata-s43
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH -e main9_fourdata_seed43_%A_%a.err
#SBATCH -o main9_fourdata_seed43_%A_%a.log

set -eo pipefail
echo "script_version=main9-fourdata-seed43-v1; task=${SLURM_ARRAY_TASK_ID:-unset}"
case "${SLURM_ARRAY_TASK_ID:-}" in
  0|1|2|3) ;;
  *) echo 'Submit: sbatch --array=0-3%4 run-Fed9.sh' >&2; exit 2 ;;
esac

export PYTHONPATH=""
export CIFAR_PATH=/gpfs/home/W125211040/FedMNPU/data
export DATASET_PATH=/gpfs/home/W125211040/FedMNPU/data
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedMNPU || exit 1

DATASETS=(fashion_mnist cifar10 chest kolektor_sdd2)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SEED=43
ROUNDS=200
MAX_STEPS=15
LOCAL_EPOCHS=6
OUTPUT_DIR="main9_fourdata_seed43_outputs/${SLURM_ARRAY_JOB_ID}/${DATASET}"
CHEST_ROOT=/gpfs/home/W125211040/FedMNPU/data/cifar-10-batches-py/chest_xray
KOLEKTOR_ROOT=/gpfs/home/W125211040/FedMNPU/data/KolektorSDD2
mkdir -p "$OUTPUT_DIR"

python - <<'PY'
import inspect
import numpy as np
import main9
import FedEPPU9

defaults = main9.build_parser().parse_args([])
options = main9.build_parser()._option_string_actions
aggregation_cls = getattr(FedEPPU9, "ComplementMassAggregation", None)
support_weight_fn = getattr(
    aggregation_cls, "support_aware_weights", None
)
probe_weights = (
    support_weight_fn(np.asarray([.4923, .0077, .5]), np.asarray([.27, 1., 1.]))
    if support_weight_fn is not None else np.asarray([])
)
checks = {
    "engine": main9.train_fedavg_proposed is FedEPPU9.train_fedavg_proposed,
    "independent loss": FedEPPU9._PU_LOSS_MODULE == "pu_loss9",
    "support function": hasattr(FedEPPU9, "linear_ranking_support"),
    "complement-mass aggregation": aggregation_cls is not None and support_weight_fn is not None,
    "support-aware normalization": (
        probe_weights.size == 3
        and abs(float(probe_weights.sum()) - 1.0) < 1e-12
        and probe_weights[0] < .4923
        and probe_weights[2] > .5
    ),
    "diagnostic objective": "diagnostics" in inspect.signature(
        FedEPPU9.ClassificationRankingPUObjective.compute
    ).parameters,
    "step cap": defaults.max_steps_per_epoch == 15,
    "local epochs": defaults.local_epochs == 6,
    "Chest training root option": "--chest_root" in options,
    "Chest validation root option": "--chest_val_root" in options,
    "Kolektor root option": "--kolektor_root" in options,
    "four datasets": all(
        name in main9._CONFIGS
        for name in ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")
    ),
    "complement-mass formula": (
        aggregation_cls is not None
        and np.allclose(
            aggregation_cls.weights(np.asarray([364., 3786., 44826.])),
            np.asarray([364., 3786., 4150.]) / 8300.,
        )
    ),
    "gradient balance removed": not hasattr(FedEPPU9, "_shared_gradient_norm"),
    "support .9986": abs(FedEPPU9.linear_ranking_support(.9986, 192) - .2688) < 1e-9,
    "support .9995": abs(FedEPPU9.linear_ranking_support(.9995, 192) - .096) < 1e-9,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main9 files on server are out of sync: " + ", ".join(failed))
print("preflight=ok; program=main9.py; datasets=fashion_mnist,cifar10,chest,kolektor_sdd2")
print("seed=43; step_cap=15; local_epochs=6; local_loss=nnpu_plus_ranking")
print("support=q=min(1,(1-prior)*batch_unlabeled_count); hard_support_gate=off")
print("aggregation=normalize(complement_mass_weight*round_mean_support); tunable_cap=none")
PY

python -m unittest -v test_main9_simplified

EXTRA_ARGS=()
if [ "$DATASET" = "chest" ]; then
  EXTRA_ARGS+=(--chest_root "$CHEST_ROOT")
  EXTRA_ARGS+=(--chest_val_root "$CHEST_ROOT")
elif [ "$DATASET" = "kolektor_sdd2" ]; then
  EXTRA_ARGS+=(--kolektor_root "$KOLEKTOR_ROOT")
fi

echo "array_task=$SLURM_ARRAY_TASK_ID; dataset=$DATASET; seed=$SEED; max_steps_per_epoch=$MAX_STEPS; local_epochs=$LOCAL_EPOCHS; rounds=$ROUNDS; local_loss=nnpu_plus_ranking; support=server_only"

srun python -u main9.py \
  --dataset "$DATASET" \
  --rounds "$ROUNDS" \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha 0.1 \
  --batch_size 256 \
  --max_steps_per_epoch "$MAX_STEPS" \
  --local_epochs "$LOCAL_EPOCHS" \
  "${EXTRA_ARGS[@]}" \
  --output "$OUTPUT_DIR/main9_${DATASET}_simplified_seed${SEED}.png"
