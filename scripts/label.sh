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
echo "TAPIP3D Annot Script"
echo "Start Time: $(date)"
echo "================================================================================"

# Environment Setup
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
echo "=> Environment Configured:"
echo "   CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"
echo "   NCCL_P2P_DISABLE:  $NCCL_P2P_DISABLE"

# The dataset to label
export provider_cfg=drivetrack_full

torchrun --nproc_per_node=$ngpus -m annotation.megasam --use_gt_intrinsics --output_path ../annotations/drivetrack_full/megasam --provider_cfg $provider_cfg