#!/bin/bash

BATCH_SIZE=12

# Explicit prompt list
PROMPTS=(
  29
  103
  175 176 177 178 179
  229
  231 232 233 234 235 236 237 238 239
  280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299
  310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329
  336
  149
  89
  174
  209
  230
  269
  279
  309
  359
  389
)

TOTAL=${#PROMPTS[@]}

i=0
while [ $i -lt $TOTAL ]; do
    # Take next batch
    BATCH=("${PROMPTS[@]:$i:$BATCH_SIZE}")

    BATCH_START=${BATCH[0]}
    BATCH_END=${BATCH[${#BATCH[@]}-1]}
    JOB_NAME="scale_test_${BATCH_START}_${BATCH_END}_instruct"

    echo "Submitting batch: ${BATCH[*]}"
    echo "Job name: $JOB_NAME"

    sbatch --job-name="$JOB_NAME" script/all_exp_scale_instruct.sh "${BATCH[@]}"

    ((i+=BATCH_SIZE))
done
