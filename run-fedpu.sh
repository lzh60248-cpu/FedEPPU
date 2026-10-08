#!/bin/bash
#SBATCH --job-name=fedpu-cifar10
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH -e fedpu_%A_%a.err
#SBATCH -o fedpu_%A_%a.log

set -eo pipefail
export PYTHONPATH=""
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU || exit 1

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
echo "method=FedPU seed=${SEED}"

python -m unittest -v test_fedpu

srun python -u fedpu_main.py \
  --dataset cifar10 \
  --rounds 100 \
  --n_clients 3 \
  --seed "${SEED}" \
  --alpha 0.1 \
  --batch_size 256 \
  --local_epochs 6 \
  --max_steps_per_epoch 15 \
  --lr 0.01 \
  --pu_weight 1.0 \
  --output "fedpu_cifar10_seed${SEED}.pt"
