#!/bin/bash
#SBATCH --job-name=bitnet_gptq_lsq_channel
#SBATCH --output=slurm/bitnet_gptq_lsq_channel_%j.out
#SBATCH --error=slurm/bitnet_gptq_lsq_channel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=a100  # Update this with your actual partition name

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"s
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"

# Load environment (adjust based on your cluster setup)
# Uncomment and modify as needed:
# module load cuda/11.8
# module load python/3.12

# Activate conda environment if using conda
# Uncomment if needed:
# source activate base  # or your conda environment name

# Set environment variables from .env file
# These are set directly here to ensure they're available to all DDP worker processes
export HF_HOME=/iridisfs/geosets/oeg1n18/huggingface
export HF_HUB_CACHE=/iridisfs/geosets/oeg1n18/huggingface
export HF_DATASETS_CACHE=/iridisfs/geosets/oeg1n18/huggingface
export BITLAB_DATA_DIR=/iridisfs/geosets/oeg1n18/bitlabdata

# Print environment info
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "BITLAB_DATA_DIR: $BITLAB_DATA_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /home/oeg1n18/BitLab

# Run training
# Using srun ensures each task gets proper GPU assignment and SLURM environment variables
echo "Starting training at $(date)"
srun python -m src.train runs/training/bitdistill/bitnet_gptq_lsq_channel.yaml

echo "Training completed at $(date)"
