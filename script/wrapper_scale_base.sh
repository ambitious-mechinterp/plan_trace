#!/bin/bash

START=11
END=80
BATCH_SIZE=10

PROMPTS=($(seq $START $END))
TOTAL=${#PROMPTS[@]}

i=0
while [ $i -lt $TOTAL ]; do
    # Extract the next batch of N prompts
    BATCH=("${PROMPTS[@]:$i:$BATCH_SIZE}")

    # Name the job using the first and last element of the batch
    BATCH_START=${BATCH[0]}
    BATCH_END=${BATCH[-1]}
    JOB_NAME="exp_scale_base_${BATCH_START}_${BATCH_END}"

    echo "Submitting batch: ${BATCH[*]} with job name: $JOB_NAME"

    # Submit SLURM job with arguments
    sbatch --job-name="$JOB_NAME" script/exp_scale_comp_base.sh "${BATCH[@]}"

    # Move to next batch
    ((i+=BATCH_SIZE))
done
