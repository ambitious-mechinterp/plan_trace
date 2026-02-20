#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=200GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 4:00:00
#SBATCH --constraint=vram40
#SBATCH -o /home/jnainani_umass_edu/w/plan_trace/logs/random_steer_scale/base/slurm_random_steer_scale.out
#SBATCH -e /home/jnainani_umass_edu/w/plan_trace/logs/random_steer_scale/base/slurm_random_steer_scale.err
#SBATCH -A pi_jensen_umass_edu

set -e

mkdir -p /home/jnainani_umass_edu/w/plan_trace/logs/random_steer_scale/base

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

module load conda/latest
conda activate finetuning

cd /home/jnainani_umass_edu/w/plan_trace/notebooks

export PYTHONUNBUFFERED=1
export RUN_MODE=base

python random_steering_check.py

echo "Job completed at: $(date)"
