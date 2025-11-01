export CLUSTER=
export CONTAINER_IMAGE=
export MEGATRON_PATH=
export MCORE_RELEASE_VERSION=
export GB200_CLUSTER=1

export MODEL=DeepSeek-V3
export WANDB_PROJECT=
export RUN_NAME="${MODEL}-benchmarking"

export BINDPCIE_PATH=${BINDPCIE_PATH:-""}

# Training parameters
export SEQ_LEN=4096
export GBS=2048
export MBS=1
export TP=1
export EP=${EP:-32}
export PP=${PP:-8}
export VPP=${VPP:-4}
export NNODES=64
export MOE_GROUPED_GEMM=true
export PRETRAIN=1
export SEGMENT=$((EP * TP / 4))
export PR=mxfp8
export DISPATCHER=${DISPATCHER:-"hybridep"}

export RUN_TIME=00:30:00

# 256 GB200 GPUs, w/o MTP, w/ force load balancing
bash sbatch_benchmarking.sh \
  --recompute-granularity selective --recompute-modules moe_act mlp \
  --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess --te-rng-tracker --cuda-graph-warmup-steps 0 \
  --moe-router-force-load-balancing \
  --pipeline-model-parallel-layout "Et|(tt|)*30L" \
