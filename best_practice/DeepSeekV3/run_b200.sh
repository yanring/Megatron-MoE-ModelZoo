# Cluster variables
export ACCOUNT=${ACCOUNT:-""}
export CONTAINER_MOUNTS="/lustre/:/lustre/"

# set the following variables
export BINDPCIE_PATH=${BINDPCIE_PATH:-""}
# export MCORE_RELEASE_VERSION=
# export MEGATRON_PATH=
# export CONTAINER_IMAGE=

# Model selection parameters
export MODEL=DeepSeek-V3
export RUN_NAME="${MODEL}-benchmarking"
export DATASET=Slimpajama
export DATA_PATH=

# Training parameters
export SEQ_LEN=4096
export GBS=${GBS:-8192}
export MBS=${MBS:-1}
export TP=${TP:-1}
export EP=${EP:-32}
export PP=${PP:-8}
export VPP=""
export NNODES=${NNODES:-32}
export MOE_GROUPED_GEMM=true
export PRETRAIN=1
export PR=mxfp8
export DISPATCHER="deepep"
export A2A_OVERLAP=1
export PROFILE=1

export RUN_TIME=00:30:00
bash sbatch_benchmarking.sh \
  --recompute-granularity selective --recompute-modules mlp mla_up_proj \
  --pipeline-model-parallel-layout "Et*4|(tttt|)*14tmL" \
  --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
  --moe-router-force-load-balancing \
  --mock-data \
  --moe-deepep-num-sms 24
