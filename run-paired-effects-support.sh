#!/bin/bash
#SBATCH --job-name=paired-support
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e paired_support_%j.err
#SBATCH -o paired_support_%j.log
set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU
python -u summarize_paired_effects_existing.py
