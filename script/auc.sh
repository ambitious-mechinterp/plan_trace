#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=200GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 12:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/slurm_auc_token.out  # %A is the master job ID, %a is the array task ID
#SBATCH -e logs/slurm_auc_token.err
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
python -m notebooks.auc_token 11 \
    --prompts-path data/prompts.json \
        --num-seqs 10000 \
        --latent-layer 7 \
        --latent-index 7643 \
        --sae-release gemma-scope-2b-pt-mlp \
        --sae-id "layer_7/width_16k/average_l0_86"

echo "Job completed at: $(date)" 
