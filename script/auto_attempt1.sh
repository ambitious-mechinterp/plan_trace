#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=200GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 12:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/slurm_auto_task11sv2.out  # %A is the master job ID, %a is the array task ID
#SBATCH -e logs/slurm_auto_task11sv2.err
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
python -m plan_trace.pipeline 11 \
  --max-tokens 1 \
  --save \
  --output-dir outputs/topkfile/ \
  --cluster-mode saved_topk \
  --cluster-saved-dir outputs/agg_per_layer_top20 \
  --cluster-saved-topk 20 \
  --k-max 90001 

echo "Job completed at: $(date)" 
