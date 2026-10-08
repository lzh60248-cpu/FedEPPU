#!/bin/bash
#SBATCH --job-name=cifar-main7-rank-cache
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e cifar10_main7_%j.err
#SBATCH -o cifar10_main7_%j.log

set -eo pipefail
export PYTHONPATH=""
export CIFAR_PATH=/gpfs/home/W125211040/FedMNPU/data
export DATASET_PATH=/gpfs/home/W125211040/FedMNPU/data
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedMNPU

SEED=42
ALPHA=0.1
ROUNDS=300
OUTPUT_DIR=main7_outputs
mkdir -p "$OUTPUT_DIR"

echo "Python Path: $(which python)"
python -c "import torch; print('Torch OK! Version:', torch.__version__)"
python - <<'PY'
import inspect
import main7
import FedEPPU7
import datasets.dataset7 as dataset

checks = {
    "main7.py": "--ranking_balance_update_interval" in main7.build_parser()._option_string_actions,
    "FedEPPU7.py": "ranking_balance_update_interval" in inspect.signature(FedEPPU7.train_fedavg_proposed).parameters,
    "pu_loss7.py": FedEPPU7._PU_LOSS_MODULE == "pu_loss7",
    "datasets/dataset7.py": "alpha" in inspect.signature(dataset.load_dataset).parameters,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Main7 files on server are out of sync: " + ", ".join(failed))
print("preflight=ok; dataset=cifar10; seed=43; alpha=0.1; rounds=300; runs=1; ranking_interval=5")
PY

srun python -u main7.py \
  --dataset cifar10 \
  --rounds "$ROUNDS" \
  --n_clients 3 \
  --seed "$SEED" \
  --alpha "$ALPHA" \
  --n_shrink 200 \
  --lambda_prior 0.10 \
  --proxy_min_specificity 0.95 \
  --threshold_objective acc \
  --batch_size 256 \
  --max_steps_per_epoch 128 \
  --local_epochs 6 \
  --ranking_balance_update_interval 5 \
  --output "$OUTPUT_DIR/main7_cifar10_alpha${ALPHA}_seed${SEED}.png"
