#!/bin/bash

BATCH_SIZE=1

# Prompts from 1 to 10
# PROMPTS=($(seq 1 10))
# PROMPTS=($(seq 11 40))
PROMPTS=($(seq 41 100))

TOTAL=${#PROMPTS[@]}

i=0
while [ $i -lt $TOTAL ]; do
    # Take next batch
    BATCH=("${PROMPTS[@]:$i:$BATCH_SIZE}")

    BATCH_START=${BATCH[0]}
    BATCH_END=${BATCH[${#BATCH[@]}-1]}
    JOB_NAME="plan_scale_test_${BATCH_START}_${BATCH_END}_instruct"

    echo "Submitting batch: ${BATCH[*]}"
    echo "Job name: $JOB_NAME"

    sbatch --job-name="$JOB_NAME" script/sec3_exp_scale.sh "${BATCH[@]}"

    ((i+=BATCH_SIZE))
done
