#!/usr/bin/env zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --mem=500g
#SBATCH --gpus=L40S:8

# Exit on error
set -e

echo "================================================================================"
echo "TAPIP3D Eval Script"
echo "Start Time: $(date)"
echo "================================================================================"

# Environment Setup
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
echo "=> Environment Configured:"
echo "   CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"
echo "   NCCL_P2P_DISABLE:  $NCCL_P2P_DISABLE"

# Hyperparameters
export lr="5e-4"
export weight_decay="5e-4"
ngpus=8
config_name=tapip3d
experiment_name="tapip3d_eval"

echo "=> Training Configuration:"
echo "   Config Name:     $config_name"
echo "   Experiment Name: $experiment_name"
echo "   Learning Rate:   $lr"
echo "   Weight Decay:    $weight_decay"
echo "   Num GPUs:        $ngpus"
echo "================================================================================"

export checkpoint="checkpoints/tapip3d_final.pth"

args=(
  wandb.project=TAPIP3D \
  train.weight_decay=$weight_decay \
  train.mixed_precision=bf16 \
  train.lr=$lr \
  train.scheduler_name=onecycle \
  train.train_steps=200000 \
  +train.eval_only=true \
  +model.eval_mode="raw" \
  train.checkpoint=$checkpoint \
  +train.visualize_with_rerun=false \
  experiment_name="$experiment_name" \
)

# Set default port if not provided
PORT=${PORT:-29500}

echo "=> Launching training with accelerate..."
if [ $ngpus -gt 1 ]; then
  accelerate launch \
    --main_process_port $PORT \
    --gpu_ids all \
    --multi_gpu \
    --num_processes $ngpus \
    train_eval.py --config-name $config_name "${args[@]}"
else
  accelerate launch \
    --main_process_port $PORT \
    --gpu_ids all \
    --num_processes $ngpus \
    train_eval.py --config-name $config_name "${args[@]}"
fi

echo "================================================================================"
echo "Eval Finished at: $(date)"
echo "================================================================================"
