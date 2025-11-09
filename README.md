# Megatron MoE Testing Guide

This guide provides detailed instructions, best practices, and optimized configurations for testing Mixtral, DeepSeek, and Qwen series models using the Megatron-Core framework to achieve optimal performance and reliability.

## Key Updates
- [DeepSeek-V3 best practices in a single command](https://github.com/yanring/Megatron-MoE-ModelZoo/blob/main/best_practice/DeepSeekV3/run.sh)
  - Current includes H100, B200, and Long Context. GB200 config coming soon.

## Table of Contents

- [Container Setup](#container-setup)
- [Design Docs](#design-docs)
- [Environment Setup](#environment-setup)
- [Performance Benchmarking](#performance-benchmarking)
- [DeepSeek Checkpoint Conversion](#deepseek-checkpoint-conversion)

# Container Setup

- **Dockerfile**: [dockers/Dockerfile](./dockers/Dockerfile)

# Design Docs
Please refer to [design_docs](./design_docs/).

# Before Running

## Login Node Setup

Before entering the container, you need to install `yq` to process `.yaml` configuration files.

<details>
<summary>Click here to view installation steps.</summary>

1.  Create a local bin directory:
    ```bash
    mkdir -p ~/.local/bin
    ```

2.  Download the `yq` executable:
    ```bash
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O ~/.local/bin/yq
    ```

3.  Make it executable:
    ```bash
    chmod +x ~/.local/bin/yq
    ```

4.  Add the local bin directory to your `PATH` in `~/.bashrc`:
    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

5.  Apply the changes:
    ```bash
    source ~/.bashrc
    ```

</details>

## Environment Setup

Before running any scripts, you need to set up the following environment variables:

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export MEGATRON_PATH="/path/to/your/megatron/directory"
export MCORE_RELEASE_VERSION="0.13"
export CONTAINER_IMAGE="/path/to/container/image.sqsh"
export CLUSTER="your_cluster_name"
```

-   **`WANDB_API_KEY`**: Your Weights & Biases API key for experiment tracking.
    -   Get your key from [wandb.ai/authorize](https://wandb.ai/authorize).
-   **`MEGATRON_PATH`**: Absolute path to your Megatron-MoE installation directory.
    -   Example: `path/to/Megatron-LM`
-   **`MCORE_RELEASE_VERSION`**: Version of Megatron-Core to use.
    -   Currently recommended: `0.13`
-   **`CONTAINER_IMAGE`**: Path to the container image file (`.sqsh`).
    -   Example: `path/to/container/image.sqsh`
-   **`CLUSTER`**: Name of your cluster environment (e.g., `EOS`, `CW`).

# Performance Benchmarking

## Benchmarking Script Usage

For performance benchmarking, you can launch scripts either with `sbatch` via [`sbatch_benchmarking.sh`](./sbatch_benchmarking.sh) or on an interactive node via [`interactive_benchmarking.sh`](./interactive_benchmarking.sh).

-   **`MODEL`**
    -   This is a required environment variable that must be set in your script or command.
    -   Predefined models include: `Mixtral-8x2B`, `Mixtral-8x7B`, `Mixtral-8x22B`, `DeepSeek-V2`, `DeepSeek-V2-Lite`, `DeepSeek-V3`, `DeepSeek-V3-Lite`, and `Qwen2-57B-A14B`.

-   **`CLUSTER`**, **`MCORE_RELEASE_VERSION`**, and **`MEGATRON_PATH`**
    -   These required variables must be defined in your script or command for proper execution.

-   **`CONTAINER_IMAGE`**

-   **Using WandB for Experiment Tracking**
    <details>
    <summary>Click here to view WandB setup instructions.</summary>

    -   To use WandB for experiment tracking, set `WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize). It is highly recommended to add `export WANDB_API_KEY="your_own_wandb_api_key"` to your `~/.bashrc`.
    -   If you do not wish to use WandB, comment out the following lines in your model's `.yaml` configuration file:
        ```yaml
        # --wandb-project: wandb_project_name
        # --wandb-exp-name: wandb_experiment_name
        ```
    </details>

## Runner Configuration Setup

All model-specific runner configurations can be adjusted through [`runtime_configs/benchmarking/runtime.conf`](./runtime_configs/benchmarking/runtime.conf) or via the benchmarking command.

-   **Available Model-Specific Runner Configurations**
    <details>
    <summary>Click here to view available model-specific benchmarking configurations.</summary>

    -   **Parallel Mappings**: `TP`, `PP`, `EP`, `CP`, `VPP`, `PP_FIRST`, `PP_LAST`, and `LAYERS_PER_VP`
    -   **Batch Sizes**: `MBS` and `GBS`
    -   **Model Architecture**: `NUM_LAYERS`
    -   **MoE Configurations**: `MOE_TOKEN_DISPATCHER`, `MOE_GROUPED_GEMM`, and `--moe-extended-ep`
    -   **Training Configurations**: `NNODES`, `RUN_TIME`, and `PRETRAIN`. Note that specifying a shorter run time may improve your job's priority in the Slurm queue.
    -   **Data Configurations**: `SEQ_LEN` and `DATASET`
    </details>
-   All available optimial configurations are listed in [`runtime_configs/benchmarking/runtime.conf`](./runtime_configs/benchmarking/runtime.conf).

## Cluster-Related Configuration Setup
All cluster configurations can be customized either through `cluster_configs/benchmarking/your_own_cluster.conf` or via the benchmarking command. For guidance on creating your own cluster configurations, refer to the template provided in [`cluster_configs/benchmarking/template.conf`](./cluster_configs/benchmarking/template.conf).
-   **Required Cluster-Specific Slurm Settings**: `ACCOUNT`, `PARTITION`, `RUN_NAME`, and `CONTAINER_MOUNTS`
-   **Required Cluster-Specific Paths**: `OUTPUT_PATH`, `DATA_PATH`, `TOKENIZER_MODEL`, and `LOAD_PATH`

## Benchmarking Script Launch

-   To benchmark a model from scratch with preconfigured parameters:
    ```bash
    # Example for DeepSeek-V3
    MODEL=DeepSeek-V3 bash ./sbatch_benchmarking.sh
    ```
-   To train a model with custom parameters:
    ```bash
    # Example for DeepSeek-V3
    MODEL=DeepSeek-V3 TP=2 PP=8 EP=64 VPP=1 PP_FIRST=8 PP_LAST=5 RUN_TIME=00:60:00 NNODES=64 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj layernorm
    ```
-   To monitor your jobs, use `squeue -u $USER` for a one-time status check or `watch -n 1 squeue -u $USER` for continuous monitoring. For detailed logging, refer to the WandB dashboard.

# DeepSeek Checkpoint Conversion

| Please try [MBridge](https://github.com/ISEEKYAN/mbridge/) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for better HF<->MCore conversion support.

### 1. Download DeepSeek-V3 Checkpoint

Download the DeepSeek-V3 checkpoint from [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3):

```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3
```

The downloaded checkpoint is in FP8 format. Run the following command to convert it to BF16 format, using this [script](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py):

```bash
python inference/fp8_cast_bf16.py --input-fp8-hf-path /your/input/fp8/hf/path --output-bf16-hf-path /your/output/bf16/hf/path
```

### 2. Convert to Megatron Legacy Checkpoint

To convert the BF16 HuggingFace checkpoint to a Megatron legacy checkpoint, execute the following command:
```
# Example for DeepSeek-V3
MODEL=DeepSeek-V3 bash ./ckpt_convert_scripts/DeepSeek-V3/convert_deepseek_v3.sh
```

### 3. Convert to Distributed Checkpoint

Finally, run this command to convert the legacy checkpoint into a distributed checkpoint:

```bash
MODEL=DeepSeek-V3 TP=1 PP=4 EP=64 VPP=1 PP_FIRST=16 PP_LAST=13 NNODES=32 LOAD_PATH=/path/to/legacy/checkpoint bash ./sbatch_benchmarking.sh --ckpt-convert-save /path/to/save/distributed/checkpoint --ckpt-convert-format torch_dist --no-save-optim
```

For reference, after conversion, the legacy checkpoint is approximately 3.4T, and the distributed checkpoint is about 1.4T.
