#!/bin/bash
#SBATCH --job-name=rank-correction
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%3
#SBATCH -e rank_correction_%A_%a.err
#SBATCH -o rank_correction_%A_%a.log
set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU

CORRECTIONS=(absolute clamp)
SEEDS=(42 43 44 45 46)
TASK=${SLURM_ARRAY_TASK_ID}
CORRECTION=${CORRECTIONS[$((TASK/5))]}
SEED=${SEEDS[$((TASK%5))]}
export PYTHONHASHSEED=$SEED

REQUIRED=(rank_correction_main.py FedEPPU_rank_correction.py test_rank_correction.py pu_loss.py datasets/dataset.py)
for FILE in "${REQUIRED[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "missing_required_file=$FILE" >&2
        exit 2
    fi
done

python -m unittest -v test_rank_correction
OUT="rank_correction_outputs/${CORRECTION}/seed${SEED}"
mkdir -p "$OUT"
echo "task=$TASK/9 correction=$CORRECTION seed=$SEED strict_determinism=true"
srun python -u rank_correction_main.py \
    --dataset cifar10 \
    --rounds 200 \
    --n_clients 3 \
    --seed "$SEED" \
    --alpha 0.1 \
    --batch_size 256 \
    --local_epochs 6 \
    --max_steps_per_epoch 15 \
    --ranking_correction "$CORRECTION" \
    --strict_determinism \
    --output "$OUT/${CORRECTION}_cifar10_seed${SEED}.png"
