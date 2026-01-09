#!/bin/bash

START=0
END=391
BATCH_SIZE=30

PROMPTS=($(seq $START $END))
TOTAL=${#PROMPTS[@]}

i=0
while [ $i -lt $TOTAL ]; do
    # Extract the next batch of N prompts
    BATCH=("${PROMPTS[@]:$i:$BATCH_SIZE}")

    # Name the job using the first and last element of the batch
    BATCH_START=${BATCH[0]}
    BATCH_END=${BATCH[-1]}
    JOB_NAME="scale_test_${BATCH_START}_${BATCH_END}_instruct"

    echo "Submitting batch: ${BATCH[*]} with job name: $JOB_NAME"

    # Submit SLURM job with arguments
    sbatch --job-name="$JOB_NAME" script/all_exp_scale_instruct.sh "${BATCH[@]}"

    # Move to next batch
    ((i+=BATCH_SIZE))
done