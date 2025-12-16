# Cluster variables: please refer to cluster_configs/benchmarking/template.conf to set up dataset and checkpointing paths.
export CONTAINER_IMAGE=
export ACCOUNT=
export MEGATRON_PATH=
export PARTITION=
export RUN_NAME="${MODEL}-benchmarking"
export CONTAINER_MOUNTS=
export CLUSTER=

# Model selection parameters
export MODEL=Mixtral-8x22B
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

# Dropless training, stable performance should be around 455TFLOPS for H100.
MODEL=Mixtral-8x22B DISPATCHER=alltoall bash ./sbatch_benchmarking.sh

# Token Drop training, perf should be around 470TFLOPS for H100.
MODEL=Mixtral-8x22B DISPATCHER=alltoall bash ./sbatch_benchmarking.sh --moe-expert-capacity-factor 1.0 --moe-pad-expert-input-to-capacity
