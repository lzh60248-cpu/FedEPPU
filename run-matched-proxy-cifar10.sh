#!/bin/bash
#SBATCH --job-name=matched-proxy
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-44%3
#SBATCH -e matched_proxy_%A_%a.err
#SBATCH -o matched_proxy_%A_%a.log
set -eo pipefail
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
PROJECT_ROOT=/gpfs/home/W125211040/FedEPPU
cd "$PROJECT_ROOT"
METHODS=(centralized_pn centralized_nnpu local_nnpu fedavg_nnpu fedprox_nnpu scaffold_nnpu fedpu mrfl_nnpu fedeppu)
SEEDS=(42 43 44 45 46)
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID/5))]}; SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID%5))]}
OUT="matched_proxy_cifar10_outputs/${METHOD}/seed${SEED}"; mkdir -p "$OUT"
REQUIRED=(other/run_matched_proxy.py other/proxy_protocol.py other/data_protocol.py other/test_matched_proxy.py rankgate_main.py FedEPPU_rankgate.py pu_loss.py)
for FILE in "${REQUIRED[@]}"; do
  if [ ! -f "$FILE" ]; then echo "missing_required_file=$FILE" >&2; exit 2; fi
done
python -m unittest -v other.test_matched_proxy other.test_baselines test_fedpu
echo "task=${SLURM_ARRAY_TASK_ID}/44 method=$METHOD seed=$SEED dataset=cifar10"
if [ "$METHOD" = fedeppu ]; then
  srun python -u rankgate_main.py --dataset cifar10 --rounds 200 --n_clients 3 --seed "$SEED" --alpha 0.1 --batch_size 256 --local_epochs 6 --max_steps_per_epoch 15 --ranking_gate hard --strict_determinism --output "$OUT/fedeppu_cifar10_seed${SEED}.png"
else
  srun python -u -m other.run_matched_proxy --method "$METHOD" --dataset cifar10 --rounds 200 --n_clients 3 --seed "$SEED" --alpha 0.1 --batch_size 256 --local_epochs 6 --max_steps_per_epoch 15 --output "$OUT/${METHOD}_cifar10_seed${SEED}.pt"
fi
