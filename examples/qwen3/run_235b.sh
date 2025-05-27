# Cluster variables
export CONTAINER_IMAGE=
export ACCOUNT=
export MEGATRON_PATH=
export PARTITION=
export RUN_NAME="${MODEL}-benchmarking"
export CONTAINER_MOUNTS=
export CLUSTER=CW

# Model selection parameters
export MODEL=Qwen3-235B-A22B
export WANDB_PROJECT=
export OUTPUT_PATH=

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=baseline


# Note: Add --moe-router-force-load-balancing for stable benchmarking
# Inter-node EP solution
TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing 

# Inter-node EP + 1F1B overlap solution
TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --combined-1f1b --combined-1f1b-recipe ep_a2a --moe-router-force-load-balancing 

## Intra-node EP + large TP solution
TP=4 PP=8 VPP=4 EP=8 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --moe-router-aux-loss-fusion

## Intra-node EP + FSDP solution
TP=1 PP=1 VPP=1 EP=8 NNODES=32 GBS=2048 MBS=4 bash ./sbatch_benchmarking.sh --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --moe-router-force-load-balancing --use-custom-fsdp --data-parallel-sharding-strategy optim_grads_params --no-gradient-accumulation-fusion --init-model-with-meta-device

## Intra-node EP + large PP solution
TP=2 PP=32 VPP=3 EP=8 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing
