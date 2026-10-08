#!/bin/bash
#SBATCH --job-name=fedeppu-hardgate
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%3
#SBATCH -e fedeppu_hardgate_%A_%a.err
#SBATCH -o fedeppu_hardgate_%A_%a.log

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=""
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

PROJECT_ROOT=/gpfs/home/W125211040/FedEPPU
cd "$PROJECT_ROOT"
export CIFAR_PATH="$PROJECT_ROOT/data"
export DATASET_PATH="$PROJECT_ROOT/data"

DATASETS=(fashion_mnist cifar10 chest kolektor_sdd2)
SEEDS=(42 43 44 45 46)
TASK=${SLURM_ARRAY_TASK_ID}
if ! [[ "$TASK" =~ ^[0-9]+$ ]] || [ "$TASK" -gt 19 ]; then
    echo "submit_with=sbatch run-Fed.sh" >&2
    exit 2
fi
DATASET=${DATASETS[$((TASK/5))]}
SEED=${SEEDS[$((TASK%5))]}
export PYTHONHASHSEED=$SEED

REQUIRED=(main.py FedEPPU.py pu_loss.py datasets/dataset.py test_main_pu.py)
for FILE in "${REQUIRED[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "missing_required_file=$FILE" >&2
        exit 2
    fi
done

python -m unittest -v test_main_pu
python - <<'PY'
import torch
import main
import FedEPPU

labels = torch.tensor([1] * 64 + [-1] * 192)
logits = torch.zeros(256, 1)
ranking = torch.zeros(256, 1)
features = torch.zeros(256, 2)

_, extreme_scale = FedEPPU.ClassificationRankingPUObjective.compute(
    logits, ranking, features, labels, .9995
)
_, supported_scale = FedEPPU.ClassificationRankingPUObjective.compute(
    logits, ranking, features, labels, .5
)
checks = {
    "engine": main.train_fedavg_proposed is FedEPPU.train_fedavg_proposed,
    "extreme client ranking disabled": float(extreme_scale.item()) == 0.0,
    "supported client ranking enabled": float(supported_scale.item()) == 1.0,
    "no model EMA": not hasattr(FedEPPU, "_update_model_ema"),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Formal FedEPPU files out of sync: " + ", ".join(failed))
print("preflight=ok; program=main.py; engine=FedEPPU.py")
print("local_objective=nnPU+I(expected_negative_count>=1)*ranking")
print("aggregation=complement_mass_times_round_mean_negative_support")
PY

EXTRA_ARGS=()
if [ "$DATASET" = "chest" ]; then
    CHEST_ROOT=""
    while IFS= read -r CHEST_PNEUMONIA_DIR; do
        CANDIDATE_ROOT="$(dirname "$(dirname "$CHEST_PNEUMONIA_DIR")")"
        if [ -d "$CANDIDATE_ROOT/train/PNEUMONIA" ] && \
           [ -d "$CANDIDATE_ROOT/train/NORMAL" ] && \
           [ -d "$CANDIDATE_ROOT/test/PNEUMONIA" ] && \
           [ -d "$CANDIDATE_ROOT/test/NORMAL" ]; then
            CHEST_ROOT="$CANDIDATE_ROOT"
            break
        fi
    done < <(find "$PROJECT_ROOT/data" -type d -path '*/train/PNEUMONIA' 2>/dev/null)
    if [ -z "$CHEST_ROOT" ]; then
        echo "invalid_chest_root: no valid Chest root under $PROJECT_ROOT/data" >&2
        exit 2
    fi
    EXTRA_ARGS+=(--chest_root "$CHEST_ROOT" --chest_val_root "$CHEST_ROOT")
elif [ "$DATASET" = "kolektor_sdd2" ]; then
    KOLEKTOR_ROOT="$PROJECT_ROOT/data/KolektorSDD2"
    if [ ! -d "$KOLEKTOR_ROOT/train" ] || [ ! -d "$KOLEKTOR_ROOT/test" ]; then
        echo "invalid_kolektor_root=$KOLEKTOR_ROOT" >&2
        exit 2
    fi
    EXTRA_ARGS+=(--kolektor_root "$KOLEKTOR_ROOT")
fi

OUT="fedeppu_hardgate_outputs/${SLURM_ARRAY_JOB_ID}/${DATASET}/seed${SEED}"
mkdir -p "$OUT"
echo "task=$TASK/19 dataset=$DATASET seed=$SEED strict_determinism=true"

srun python -u main.py \
    --dataset "$DATASET" \
    --rounds 200 \
    --n_clients 3 \
    --seed "$SEED" \
    --alpha 0.1 \
    --batch_size 256 \
    --local_epochs 6 \
    --max_steps_per_epoch 15 \
    --strict_determinism \
    "${EXTRA_ARGS[@]}" \
    --output "$OUT/fedeppu_${DATASET}_seed${SEED}.png"
