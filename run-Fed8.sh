#!/bin/bash
#SBATCH --job-name=chest-main8-from5
#SBATCH --constraint="Python"
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -e chest_main8_%j.err
#SBATCH -o chest_main8_%j.log

set -eo pipefail
export PYTHONPATH=""

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125211040/anaconda3/bin/activate
conda activate pu_env
cd /gpfs/home/W125211040/FedMNPU
CHEST_ROOT=/gpfs/home/W125211040/FedMNPU/data/cifar-10-batches-py/chest_xray
if [[ ! -d "$CHEST_ROOT/train" || ! -d "$CHEST_ROOT/test" ]]; then
    echo "Chest dataset train/ or test/ directory is missing under $CHEST_ROOT" >&2
    exit 1
fi

echo "Python Path: $(which python)"
python -c "import torch; print('Torch OK! Version:', torch.__version__)"
python - <<'PY'
import inspect
import main8
import FedEPPU8
import datasets.dataset8 as dataset8

if main8.EXPERIMENT_ID != "main8_from_main5_chest_v1":
    raise SystemExit("Server main8.py is outdated; upload the matching Main8 files.")
if "chest_root" not in inspect.signature(dataset8.load_dataset).parameters:
    raise SystemExit("Server datasets/dataset8.py does not accept --chest_root.")
if FedEPPU8.pu_loss.__module__ != "pu_loss8":
    raise SystemExit("Server FedEPPU8.py is not paired with pu_loss8.py.")
print("preflight=ok; dataset=chest; seed=42; alpha=0.1; runs=1")
PY

srun python -u main8.py \
    --dataset chest \
    --chest_root "$CHEST_ROOT" \
    --rounds 200 \
    --n_clients 3 \
    --seed 42 \
    --alpha 0.1 \
    --n_shrink 200 \
    --lambda_prior 0.10 \
    --batch_size 256 \
    --max_steps_per_epoch 128 \
    --local_epochs 6
