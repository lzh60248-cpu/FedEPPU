#!/bin/bash
#SBATCH --job-name=sum-matched
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e summarize_matched_proxy_%j.err
#SBATCH -o summarize_matched_proxy_%j.log
set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU
python -u summarize_matched_proxy.py --root matched_proxy_cifar10_outputs --output matched_proxy_cifar10_summary.csv
