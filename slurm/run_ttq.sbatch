#!/bin/bash
# MLPerf Training Llama2-70B LoRA time-to-quality (TTQ) run on GH200 nodes.
# Use ./submit.sh instead of calling sbatch directly; it sets -N, partition, comment
# and the experiment overrides.
#
# Configuration source:
#   * KRAI v6.0 Isambard-AI config for the node count if it exists (n1,n2,n4,n8,n16),
#     otherwise KRAI's common config_gh200.sh (MINIBS=8, CP=1) -> pass overrides.
#   * Overrides (from submit.sh):  OV_MINIBS OV_CP OV_LR OV_MAX_STEPS OV_CP_COMM_TYPE
#   * NETMODE=C (default): NCCL over Slingshot RDMA via host Cray libfabric (CXI)
#     NETMODE=A          : image as-is -> NCCL falls back to TCP sockets
#SBATCH -J kv6-ttq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH -o results/runs/%x-%j.out

set -x
source "$SLURM_SUBMIT_DIR/env.sh"
NN=$SLURM_JOB_NUM_NODES
NETMODE=${NETMODE:-C}

# ---- KRAI configuration -----------------------------------------------------
KDIR=$KRAI_IMPL/Isambard-AI-Phase2-GH200-96GB_n${NN}_v6.0
if [ -f "$KDIR/config_gh200_n${NN}_tp4pp1cp1.sh" ]; then
  KCFG=$KDIR; CFGF=config_gh200_n${NN}_tp4pp1cp1.sh
else
  KCFG=$KRAI_IMPL/Isambard-AI-Phase2-GH200-96GB_n16_v6.0; CFGF=config_gh200.sh
fi
cd "$KCFG" && source "./$CFGF"
export DGXNNODES=$NN
for v in MINIBS CP LR MAX_STEPS CP_COMM_TYPE; do
  eval "[ -n \"\${OV_$v:-}\" ] && export $v=\${OV_$v}"
done
echo "CP_COMM_TYPE=${CP_COMM_TYPE:-a2a(default)}"

TAG=n${NN}-mb${MINIBS}-cp${CP}${CP_COMM_TYPE:+-$CP_COMM_TYPE}-lr${LR}-ms${MAX_STEPS}-net${NETMODE}
LOGDIR=$RESULTS/$TAG-$SLURM_JOB_ID
mkdir -p "$LOGDIR/data_index/train" "$LOGDIR/data_index/val"
echo "EXPTAG=$TAG"

# ---- Runtime environment ------------------------------------------------------
export SEED=${SEED:-$RANDOM}
export OMP_NUM_THREADS=$((CPUS_PER_TASK / DGXNGPU))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PYTHONPATH=/workspace/ft-llm/Megatron-LM
# Node-local compile caches: a Triton cache shared on Lustre by 256 ranks caused a
# "KeyError: 'cubin'" race and a hang before warmup at 64 nodes.
export TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
# Pyxis rootfs is read-only by default; keep HF/XDG caches on the results mount.
export HF_HOME=/results/.cache/huggingface XDG_CACHE_HOME=/results/.cache HOME=/results
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
env | grep -E '^(MINIBS|CP|TP|PP|LR|MAX_STEPS|DGXNNODES|SEED|FP8|MCORE_CUDA_GRAPH)=' | sort

MNT=$DATADIR:/data:ro,$MODEL:/ckpt:ro,$LOGDIR:/results:rw,$KCFG/train.py:/workspace/ft-llm/train.py:ro

if [ "$NETMODE" = "C" ]; then
  # Collect host libfabric dependencies once (skip glibc-family libs so the
  # container's own system libraries are not shadowed).
  if [ ! -f "$HOSTLIBS/.done" ]; then
    mkdir -p "$HOSTLIBS"
    ldd "$CRAY_LIBFABRIC/lib64/libfabric.so.1" | awk '/=> \//{print $3}' \
      | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux|libgcc_s|libstdc\+\+)[.-]' \
      | xargs -r -I{} cp -L {} "$HOSTLIBS/" && touch "$HOSTLIBS/.done"
  fi
  MNT=$MNT,/opt/cray:/opt/cray:ro,$HOSTLIBS:/hostlibs:ro
  export NETENV="export LD_LIBRARY_PATH=$CRAY_LIBFABRIC/lib64:/hostlibs:\$LD_LIBRARY_PATH \
    NCCL_NET_PLUGIN=aws-ofi NCCL_NET='AWS Libfabric' FI_PROVIDER=cxi \
    FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd \
    FI_CXI_DEFAULT_CQ_SIZE=131072 NCCL_CROSS_NIC=1"
else
  export NETENV=':'
fi

# ---- Launch: 1 Slurm task per node -> torchrun spawns 4 ranks per node ------------
srun --ntasks-per-node=1 --mpi=none \
  --container-image="$IMG" --container-writable \
  --container-mounts="$MNT" \
  --container-workdir=/workspace/ft-llm \
  bash -c 'export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:$PATH
           eval "$NETENV"
           slurm2pytorch torchrun --nproc_per_node=${DGXNGPU} --nnodes=${DGXNNODES} \
             --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
             --rdzv_id=kv6_${SLURM_JOB_ID} \
             --rdzv_conf=read_timeout=600,join_timeout=300 train.py' \
  2>&1 | tee "$LOGDIR/run.log"
echo "RUN_EXIT=${PIPESTATUS[0]}"
