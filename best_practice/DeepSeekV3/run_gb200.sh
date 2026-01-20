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
export GBS=${GBS:-8192}
export MBS=${MBS:-1}
export TP=${TP:-1}
export EP=${EP:-32}
export PP=${PP:-8}
export VPP=${VPP:-4}
export NNODES=${NNODES:-64}
export MOE_GROUPED_GEMM=true
export PRETRAIN=${PRETRAIN:-1}
export SEGMENT=$((EP * TP / 4))
export PR=mxfp8
export DISPATCHER=${DISPATCHER:-"hybridep"}

export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$((EP * TP))
export USE_MNNVL=1
export NUM_OF_STAGES_DISPATCH_API=10
export NUM_OF_IN_FLIGHT_S2G_DISPATCH_API=8

export RUN_TIME=${RUN_TIME:-"00:20:00"}

# 256 GB200 GPUs, w/ MTP, w/ force load balancing
bash sbatch_benchmarking.sh \
  --recompute-granularity selective --recompute-modules mlp \
  --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess --te-rng-tracker --cuda-graph-warmup-steps 0 \
  --pipeline-model-parallel-layout "Et*4|(tttt|)*14tmL" \
  --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
  --offload-optimizer-states \
  --moe-router-force-load-balancing \
