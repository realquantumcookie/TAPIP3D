# Dataset Preparation

## Configuration

The project configuration consists of four main components:

- `scripts/`: Shell scripts for training and evaluation, specifying the main configuration.
- `configs/xxx.yaml`: Main configuration files (e.g., `tapip3d.yaml`) defining the datasets for training and evaluation.
- `configs/dataset/`: Dataset-specific configurations defining dataset paths and transformations.
- `configs/annotation/`: Optional configurations for estimated depth, specifying paths to predicted depth maps.

Dataset preparation involves configuring the main experiment file and setting local paths for data and annotations.

We provide configuration settings for all datasets and annotations used in the paper's main tables. However, users must download the data and configure the root paths for datasets and annotations.

All evaluation datasets are listed at the top of `configs/tapip3d.yaml` (commented out). Please uncomment the relevant lines after downloading the data.

## Download the Dataset

### 1. Kubric MOVi-F (Training Dataset)
The exact training dataset used in the paper is available at https://huggingface.co/datasets/zbww/tapip3d-kubric.

### 2. LSFOdyssey
You may download the dataset from https://huggingface.co/datasets/wwcreator/LSFOdyssey.

### 3. DexYCB
Follow the instructions at https://dex-ycb.github.io/ to download the dataset.

### 4. TAPVid-3D
Due to licensing restrictions, we cannot redistribute the dataset. Please follow the instructions at https://tapvid3d.github.io/ to download and preprocess the data.

To evaluate TAPVid-3D using MegaSam estimated depth, please download our depth annotations:
https://huggingface.co/datasets/zbww/tapvid3d_megasam_oldimpl

**Note:** To reproduce our results using the provided annotations, ensure you use the TAPVid-3D dataset at commit `0497454`. The dataset has since been updated; using a different version may require regenerating annotation files.

Please note that these depth maps were generated using a slightly different script than the one released, but they will reproduce the results reported in the paper.

## Generating Annotation Files

To generate annotation files, first download the original dataset and configure its path in `annotation/provider_configs`.

Then, specify the provider configuration and execute the labeling script: `scripts/label.sh`.

Enable `--use_gt_intrinsics` for the TAPVid dataset. For other datasets without intrinsic parameters, this option must be disabled.
