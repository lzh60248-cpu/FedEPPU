#!/bin/bash
#SBATCH --job-name=sum-ctrl-prior
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e summarize_controlled_prior_%j.err
#SBATCH -o summarize_controlled_prior_%j.log
set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU
python -u summarize_controlled_prior.py
