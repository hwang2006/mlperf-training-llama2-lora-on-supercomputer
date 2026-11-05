#!/bin/bash
# One MLPerf Training Llama2-70B LoRA run (time to quality) on N GH200 nodes.
# Submit with ./submit.sh (sets -N, partition, comment, exclusions, config and seed).
#
# Configuration:
#   * CFG=<file in configs/>  Hangang config (required for 32 and 64 nodes; submit.sh picks it)
#   * otherwise Krai's v6.0 Isambard-AI config for the node count (1, 2, 4, 8, 16 nodes)
#   * OV_MINIBS OV_CP OV_LR OV_MAX_STEPS OV_CP_COMM_TYPE: experiment overrides. They are NOT
#     reflected in the logged config_filename, so do not use them for official runs.
#   * NETMODE=C (default): NCCL over Slingshot RDMA via the host HPE Cray libfabric (CXI)
#     NETMODE=A          : image as-is, NCCL falls back to TCP sockets
#     Single-node runs never use the Slingshot settings (no inter-node traffic, no VNI).
#SBATCH -J kv6-ttq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH -o results/runs/%x-%j.out

set -x
source "$SLURM_SUBMIT_DIR/env.sh"
NN=$SLURM_JOB_NUM_NODES
NETMODE=${NETMODE:-C}

# ---- Configuration -----------------------------------------------------------------
KDIR=$KRAI_IMPL/Isambard-AI-Phase2-GH200-96GB_n${NN}_v6.0
TRAIN_PY=$KRAI_IMPL/Isambard-AI-Phase2-GH200-96GB_n16_v6.0/train.py   # identical in all Krai dirs
if [ -n "${CFG:-}" ]; then
  CFGDIR=$CONFIGS
elif [ -f "$KDIR/config_gh200_n${NN}_tp4pp1cp1.sh" ]; then
  CFGDIR=$KDIR; CFG=config_gh200_n${NN}_tp4pp1cp1.sh
else
  echo "ERROR: no Krai config for $NN nodes; pass CFG=<file in configs/>"; exit 1
fi
cd "$CFGDIR" && source "./$CFG"
export DGXNNODES=$NN
for v in MINIBS CP LR MAX_STEPS CP_COMM_TYPE; do
  eval "if [ -n \"\${OV_$v:-}\" ]; then export $v=\${OV_$v}; echo \"OVERRIDE $v=\${OV_$v}\"; fi"
done
echo "CONFIG=$CFGDIR/$CFG"
echo "CP_COMM_TYPE=${CP_COMM_TYPE:-a2a(default)}"

TAG=n${NN}-mb${MINIBS}-cp${CP}${CP_COMM_TYPE:+-$CP_COMM_TYPE}-lr${LR}-ms${MAX_STEPS}
LOGDIR=$RESULTS/$TAG-$SLURM_JOB_ID
mkdir -p "$LOGDIR/data_index/train" "$LOGDIR/data_index/val"
echo "EXPTAG=$TAG"

# ---- Runtime environment ----------------------------------------------------------
export SEED=${SEED:-$RANDOM}
export OMP_NUM_THREADS=$((CPUS_PER_TASK / DGXNGPU))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PYTHONPATH=/workspace/ft-llm/Megatron-LM
# Node-local compile caches (a Triton cache shared on Lustre hung a 64-node run).
export TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
# Pyxis rootfs is read-only by default; keep HF/XDG caches on the results mount.
export HF_HOME=/results/.cache/huggingface XDG_CACHE_HOME=/results/.cache HOME=/results
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# In-container data/checkpoint paths read by the NeMo config (oc.env:DATA_ROOT / oc.env:CKPT_ROOT).
# Set explicitly so no host-side variable of the same name can leak in through Pyxis.
export DATA_ROOT=/data CKPT_ROOT=/ckpt
env | grep -E '^(MINIBS|MBS|CP|TP|PP|LR|MAX_STEPS|DGXNNODES|SEED|FP8|MCORE_CUDA_GRAPH)=' | sort

MNT=$DATADIR:/data:ro,$MODEL:/ckpt:ro,$LOGDIR:/results:rw,$TRAIN_PY:/workspace/ft-llm/train.py:ro

if [ "$NETMODE" = "C" ] && [ "$NN" -gt 1 ]; then
  # Host libfabric dependencies (glibc-family libs excluded so the container's are kept).
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

# ---- Launch -----------------------------------------------------------------------
# Everything inside { } goes to run.log, which becomes the MLPerf result file:
#   1. v6.1 disclosure keys + cache_clear   2. per-node page-cache size   3. training
cd "$SLURM_SUBMIT_DIR"
{
  python3 "$REPO_DIR/tools/log_disclosure.py" "$TP" "${PP:-1}" "$CP" "${MBS:-1}" "$CFG"
  srun --ntasks-per-node=1 "$REPO_DIR/tools/cache_state.sh"
  srun --ntasks-per-node=1 --mpi=none \
    --container-image="$IMG" --container-writable \
    --container-mounts="$MNT" \
    --container-workdir=/workspace/ft-llm \
    bash -c 'export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:$PATH
             eval "$NETENV"
             slurm2pytorch torchrun --nproc_per_node=${DGXNGPU} --nnodes=${DGXNNODES} \
               --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
               --rdzv_id=kv6_${SLURM_JOB_ID} \
               --rdzv_conf=read_timeout=600,join_timeout=300 train.py'
} 2>&1 | tee "$LOGDIR/run.log"
echo "RUN_EXIT=${PIPESTATUS[0]}"
