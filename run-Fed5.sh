#!/bin/bash
#SBATCH --job-name=fedeppu5-d9-200r
#SBATCH --constraint="Python"
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e %j.err
#SBATCH -o %j.log

export PYTHONPATH=""

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env

echo "Python Path: $(which python)"
python -c "import torch; print('Torch OK! Version:', torch.__version__)"

srun python -u main5.py \
    --dataset chest \
    --chest_root "/gpfs/home/W125211040/FedMNPU/data/cifar-10-batches-py/chest_xray" \
    --rounds 200 \
    --n_clients 3 \
    --seed 42 \
    --alpha 0.1 \
    --n_shrink 200 \
    --lambda_prior 0.10 \
    --batch_size 256 \
    --max_steps_per_epoch 128 \
    --local_epochs 6
