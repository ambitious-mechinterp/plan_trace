#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=200GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 4:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/all_scale/base/slurm_scale_test.out
#SBATCH -e logs/all_scale/base/slurm_scale_test.err
#SBATCH -A pi_jensen_umass_edu

# Create logs directory if it doesn't exist
mkdir -p logs/all_scale/base

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load any necessary modules (adjust as needed)
module load conda/latest
conda activate finetuning

PROMPTS=("$@")

for PROMPT in "${PROMPTS[@]}"; do
  echo "Running prompt ${PROMPT} with saved_topk_fast clustering"
  python -m plan_trace.pipeline ${PROMPT} \
    --max-tokens 2 \
    --save \
    --output-dir outputs/all_scale/base \
    --cluster-mode saved_topk_fast \
    --cluster-saved-topk-fast-cache cache/lt_cache_dir_win5_top20_match15 \
    --k-max 70001 \
    --per-position \
    --use-pos-info \
    --pos-info-offset 1 \
    --data-path data/external/all_examples_og_prompt_with_position_info.json \
    --model gemma-2-2b \
    --include-docstrings
done

echo "Job completed at: $(date)"