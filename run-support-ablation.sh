#!/bin/bash
#SBATCH --job-name=support-ablation
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%3
#SBATCH -e support_ablation_%A_%a.err
#SBATCH -o support_ablation_%A_%a.log
set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU
VARIANTS=(no_support local_only global_only local_global);SEEDS=(42 43 44 45 46)
TASK=${SLURM_ARRAY_TASK_ID};VARIANT=${VARIANTS[$((TASK/5))]};SEED=${SEEDS[$((TASK%5))]}
export PYTHONHASHSEED=$SEED
REQUIRED=(support_ablation_main.py FedEPPU_support_ablation.py test_support_ablation.py pu_loss.py datasets/dataset.py)
for FILE in "${REQUIRED[@]}";do if [ ! -f "$FILE" ];then echo "missing_required_file=$FILE" >&2;exit 2;fi;done
python -m unittest -v test_support_ablation
OUT="support_ablation_outputs/${VARIANT}/seed${SEED}";mkdir -p "$OUT"
echo "task=$TASK/19 variant=$VARIANT seed=$SEED strict_determinism=true"
srun python -u support_ablation_main.py --dataset cifar10 --rounds 200 --n_clients 3 --seed "$SEED" --alpha 0.1 --batch_size 256 --local_epochs 6 --max_steps_per_epoch 15 --variant "$VARIANT" --strict_determinism --output "$OUT/${VARIANT}_cifar10_seed${SEED}.png"
