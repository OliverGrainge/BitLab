#!/bin/bash
# Defaults are set here but overridden by submit_jobs.sh via sbatch flags.
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=a100

# --- Sanity check ---
if [[ -z "$YAML_PATH" ]]; then
    echo "ERROR: YAML_PATH is not set. Use submit_jobs.sh to launch this script."
    exit 1
fi

# --- Job info ---
echo "Job ID:            $SLURM_JOB_ID"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Node:              $SLURM_NODELIST"
echo "GPUs:              $SLURM_GPUS_ON_NODE"
echo "Workers:           $NUM_WORKERS"
echo "Config:            $YAML_PATH"
echo "Working Directory: $(pwd)"
echo "Start Time:        $(date)"

# --- Environment ---
export WANDB_MODE=offline
export HF_HOME=/iridisfs/geosets/oeg1n18/huggingface
export HF_HUB_CACHE=/iridisfs/geosets/oeg1n18/huggingface
export HF_DATASETS_CACHE=/iridisfs/geosets/oeg1n18/huggingface
export BITLAB_DATA_DIR=/iridisfs/geosets/oeg1n18/bitlabdata

echo "HF_HOME:           $HF_HOME"
echo "BITLAB_DATA_DIR:   $BITLAB_DATA_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Run ---
cd /home/oeg1n18/BitLab

echo "Starting training at $(date)"
srun python -m src.train "$YAML_PATH"
echo "Training completed at $(date)"