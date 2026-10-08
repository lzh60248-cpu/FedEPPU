#!/usr/bin/env bash
#SBATCH --job-name=prior-fedavg
#SBATCH --constraint=Python
#SBATCH -N 1
#SBATCH -p GPUFEE04
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%3
#SBATCH -o prior-fedavg_%A_%a.log
#SBATCH -e prior-fedavg_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/gpfs/home/W125211040/FedEPPU
PYTHON_BIN=/gpfs/home/W125211040/anaconda3/envs/pu_env/bin/python
PROGRAM=${PROJECT_ROOT}/fedavg_nnpu_standalone.py

if [[ ! -f "${PROGRAM}" ]]; then
    echo "missing_required_file=${PROGRAM}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing_python=${PYTHON_BIN}" >&2
    exit 2
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
LEVELS=(mild moderate extreme)
LEVEL=${LEVELS[$((TASK_ID / 5))]}
SEED=$((42 + TASK_ID % 5))
JOB_GROUP=${SLURM_ARRAY_JOB_ID:-manual}
OUTPUT_DIR=${PROJECT_ROOT}/controlled_prior_outputs/${JOB_GROUP}/${LEVEL}/fedavg_nnpu/seed${SEED}
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
echo "program=fedavg_nnpu_standalone.py protocol=controlled_prior level=${LEVEL} seed=${SEED}"

"${PYTHON_BIN}" -u "${PROGRAM}" \
    --dataset cifar10 \
    --protocol controlled_prior \
    --prior_level "${LEVEL}" \
    --positive_classes 0,1,2,3,4 \
    --num_clients 3 \
    --client_fraction 1.0 \
    --controlled_labeled_positive_per_client 333 \
    --controlled_unlabeled_per_client 15640 \
    --rounds 200 \
    --local_epochs 6 \
    --max_steps_per_epoch 15 \
    --batch_size 256 \
    --positive_batch_fraction 0.25 \
    --learning_rate 0.0015 \
    --momentum 0.0 \
    --weight_decay 0.0005 \
    --gradient_clip 1.0 \
    --eval_batch_size 2048 \
    --checkpoint_interval 10 \
    --amp \
    --seed "${SEED}" \
    --partition_seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "fedavg_nnpu_prior_${LEVEL}_seed${SEED}"
