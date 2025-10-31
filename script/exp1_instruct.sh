#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=200GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 12:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/slurm_auto_base.out
#SBATCH -e logs/slurm_auto_base.err
#SBATCH -A pi_jensen_umass_edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load any necessary modules (adjust as needed)
module load conda/latest
conda activate finetuning

# Prompts to run for the instruct model
PROMPTS=(11 15 24)

for PROMPT in "${PROMPTS[@]}"; do
  echo "Running prompt ${PROMPT} from data file and with instruct model including docstring"
  python -m plan_trace.pipeline ${PROMPT} \
    --max-tokens 50 \
    --save \
    --include-docstring \
    --output-dir outputs/exp1/instruct \
    --cluster-mode saved_topk \
    --cluster-saved-dir outputs/agg_per_layer_top20 \
    --cluster-saved-topk 20 \
    --k-max 70001
done

echo "Job completed at: $(date)"


