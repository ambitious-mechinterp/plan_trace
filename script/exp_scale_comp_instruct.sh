#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=200GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 12:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/scale_exp/instruct/slurm_auto_base.out
#SBATCH -e logs/scale_exp/instruct/slurm_auto_base.err
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

PROMPTS=("$@")

for PROMPT in "${PROMPTS[@]}"; do
  echo "Running prompt ${PROMPT} from selected file with instruct model (gemma-2-2b-it)"
  python -m plan_trace.pipeline ${PROMPT} \
    --save \
    --output-dir outputs/scale-exp/instruct \
    --cluster-mode saved_topk \
    --cluster-saved-dir outputs/agg_per_layer_top20 \
    --cluster-saved-topk 20 \
    --data-path data/external/first_100_selected_examples_without_docstrings_base_model_og_prompt_V2_instruct_comparison_with_position_info.json \
    --no-docstring \
    --use-pos-info \
    --pos-info-offset 2\
    --k-max 70001
done

echo "Job completed at: $(date)"


