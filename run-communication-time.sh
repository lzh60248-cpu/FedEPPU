#!/bin/bash
#SBATCH --job-name=comm-time
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-119%3
#SBATCH -e communication_time_%A_%a.err
#SBATCH -o communication_time_%A_%a.log

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [ "$TASK_ID" -gt 119 ]; then
  echo "Submit with: sbatch run-communication-time.sh" >&2
  exit 2
fi

DATASETS=(fashion_mnist cifar10 chest kolektor_sdd2)
METHODS=(fedavg_nnpu fedprox_nnpu scaffold_nnpu fedpu mrfl_nnpu fedeppu)
SEEDS=(42 43 44 45 46)

DATASET_INDEX=$((TASK_ID / 30))
WITHIN_DATASET=$((TASK_ID % 30))
METHOD_INDEX=$((WITHIN_DATASET / 5))
SEED_INDEX=$((WITHIN_DATASET % 5))
DATASET="${DATASETS[$DATASET_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

OUTPUT_DIR="communication_time_outputs/${SLURM_ARRAY_JOB_ID}/${DATASET}/${METHOD}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"

for REQUIRED_FILE in communication_time_main.py FedEPPU_timing.py pu_loss.py \
  datasets/dataset.py other/run_timing_baselines.py; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "missing_required_file=$REQUIRED_FILE" >&2
    exit 2
  fi
done

KOLEKTOR_ROOT="$PROJECT_ROOT/data/KolektorSDD2"
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
    echo "invalid_chest_root: no directory containing train/{NORMAL,PNEUMONIA} and test/{NORMAL,PNEUMONIA} was found under $PROJECT_ROOT/data" >&2
    exit 2
  fi
  echo "resolved_chest_root=$CHEST_ROOT"
  EXTRA_ARGS+=(--chest_root "$CHEST_ROOT" --chest_val_root "$CHEST_ROOT")
elif [ "$DATASET" = "kolektor_sdd2" ]; then
  EXTRA_ARGS+=(--kolektor_root "$KOLEKTOR_ROOT")
fi

echo "section=4.5; task=$TASK_ID/119; dataset=$DATASET; method=$METHOD; seed=$SEED"
echo "timing=cuda_synchronized_train_plus_server_aggregation; warmup_rounds=10"

if [ "$METHOD" = "fedeppu" ]; then
  srun python -u communication_time_main.py \
    --dataset "$DATASET" --rounds 200 --n_clients 3 --seed "$SEED" --alpha 0.1 \
    --batch_size 256 --max_steps_per_epoch 15 --local_epochs 6 \
    "${EXTRA_ARGS[@]}" --output "$OUTPUT_DIR/fedeppu_${DATASET}_seed${SEED}.png"
else
  srun python -u -m other.run_timing_baselines \
    --method "$METHOD" --dataset "$DATASET" --rounds 200 --n_clients 3 \
    --seed "$SEED" --alpha 0.1 --batch_size 256 \
    --max_steps_per_epoch 15 --local_epochs 6 \
    "${EXTRA_ARGS[@]}" --output "$OUTPUT_DIR/${METHOD}_${DATASET}_seed${SEED}.pt"
fi

echo "completed dataset=$DATASET method=$METHOD seed=$SEED output=$OUTPUT_DIR"
