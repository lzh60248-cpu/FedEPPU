#!/bin/bash
set -eo pipefail

TRAIN_JOB_ID="$(sbatch --parsable run-main2-local-losses.sh)"
PLOT_JOB_ID="$(sbatch --parsable --dependency=afterok:${TRAIN_JOB_ID} plot-main2-local-losses.sh "${TRAIN_JOB_ID}")"

echo "training_array_job=${TRAIN_JOB_ID}; tasks=5; max_concurrent=3"
echo "plot_job=${PLOT_JOB_ID}; starts_after_training=afterok"
echo "final_figure=main2_local_loss_outputs/${TRAIN_JOB_ID}/figure2_local_losses.pdf"
