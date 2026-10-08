#!/bin/bash
#SBATCH --job-name=ctrl-prior
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-44%3
#SBATCH -e controlled_prior_%A_%a.err
#SBATCH -o controlled_prior_%A_%a.log
set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedEPPU
CONFIGS=(mild moderate extreme);METHODS=(fedavg_nnpu ranking_fedavg fedeppu);SEEDS=(42 43 44 45 46)
TASK=${SLURM_ARRAY_TASK_ID};CONFIG=${CONFIGS[$((TASK/15))]};WITHIN=$((TASK%15));METHOD=${METHODS[$((WITHIN/5))]};SEED=${SEEDS[$((WITHIN%5))]}
REQUIRED=(controlled_prior_heterogeneity.py test_controlled_prior_heterogeneity.py FedEPPU_rankgate.py pu_loss.py other/proxy_protocol.py other/fedavg_nnpu.py other/nnpu_common.py)
for FILE in "${REQUIRED[@]}";do if [ ! -f "$FILE" ];then echo "missing_required_file=$FILE" >&2;exit 2;fi;done
python -m unittest -v test_controlled_prior_heterogeneity other.test_matched_proxy
OUT="controlled_prior_outputs/${CONFIG}/${METHOD}/seed${SEED}";mkdir -p "$OUT"
echo "task=$TASK/44 config=$CONFIG method=$METHOD seed=$SEED"
srun python -u controlled_prior_heterogeneity.py --method "$METHOD" --prior_config "$CONFIG" --seed "$SEED" --rounds 200 --batch_size 256 --local_epochs 6 --max_steps_per_epoch 15 --output "$OUT/${METHOD}_${CONFIG}_seed${SEED}.pt"
