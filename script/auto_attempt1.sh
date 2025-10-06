#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=100GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -t 8:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/slurm_auto_task24s.out  # %A is the master job ID, %a is the array task ID
#SBATCH -e logs/slurm_auto_task24s.err
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

# Navigate to the notebooks directory and run the script
python -m plan_trace.pipeline 24 \
  --max-tokens 2 \
  --save \
  --output-dir outputs/topk/ \
  --cluster-mode neuronpedia_topk \
  --cluster-api-model gemma-2-2b \
  --cluster-api-topk 20 \
  --cluster-api-source '{layer}-gemmascope-mlp-16k'

echo "Job completed at: $(date)" 
