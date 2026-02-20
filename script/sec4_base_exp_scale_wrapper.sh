#!/bin/bash

BATCH_SIZE=1

# PROMPTS=($(seq 1 100))
# PROMPTS=($(seq 1 10))
PROMPTS=($(seq 11 100))

TOTAL=${#PROMPTS[@]}

i=0
while [ $i -lt $TOTAL ]; do
    # Take next batch
    BATCH=("${PROMPTS[@]:$i:$BATCH_SIZE}")

    BATCH_START=${BATCH[0]}
    BATCH_END=${BATCH[${#BATCH[@]}-1]}
    JOB_NAME="plan_scale_test_${BATCH_START}_${BATCH_END}_base"

    echo "Submitting batch: ${BATCH[*]}"
    echo "Job name: $JOB_NAME"

    sbatch --job-name="$JOB_NAME" script/sec4_base_exp_scale.sh "${BATCH[@]}"

    ((i+=BATCH_SIZE))
done
