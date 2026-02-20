#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=200GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 4:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/planning_scale/base/slurm_scale_test.out
#SBATCH -e logs/planning_scale/base/slurm_scale_test.err
#SBATCH -A pi_jensen_umass_edu

# Create logs directory if it doesn't exist
mkdir -p logs/planning_scale/base

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
  source /work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/.venv/bin/activate
  python -m plan_trace.pipeline ${PROMPT} \
    --max-tokens 50 \
    --save \
    --output-dir outputs/planning_scale/base \
    --cluster-mode saved_topk_fast \
    --cluster-saved-topk-fast-cache cache/lt_cache_dir_win5_top20_match15 \
    --k-max 70001 \
    --per-position \
    --data-path data/external/all_examples_og_prompt_with_position_info.json \
    --per-position-coeff-start -200 \
    --per-position-coeff-end 0 \
    --per-position-coeff-step 25 \
    --model gemma-2-2b \
    --include-docstrings 
done

echo "Job completed at: $(date)"