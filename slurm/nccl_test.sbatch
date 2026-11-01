#!/bin/bash
# Compare NCCL inter-node transports inside the benchmark image (2 nodes, 8 GPUs):
#   A_default    image as-is            -> NET/Socket over bond0 + hsn*
#   B_socket_hsn NCCL_SOCKET_IFNAME=hsn -> NET/Socket over Slingshot IP only
#   C_ofi_cxi    host Cray libfabric    -> aws-ofi-nccl + CXI provider (RDMA)
# Submit from the repository root:
#   sbatch -N 2 -p normal --comment="field=efficientai;appl=megatron-lm" slurm/nccl_test.sbatch
#SBATCH -J kv6-nccl
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -o results/runs/%x-%j.out

source "$SLURM_SUBMIT_DIR/env.sh"
R=$RESULTS/nccl-$SLURM_JOB_ID; mkdir -p "$R" "$HOSTLIBS"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29600

if [ ! -f "$HOSTLIBS/.done" ]; then
  ldd "$CRAY_LIBFABRIC/lib64/libfabric.so.1" | awk '/=> \//{print $3}' \
    | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux|libgcc_s|libstdc\+\+)[.-]' \
    | xargs -r -I{} cp -L {} "$HOSTLIBS/" && touch "$HOSTLIBS/.done"
fi

echo "== Slingshot step env"
srun -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -c 'echo $(hostname) VNIS=$SLINGSHOT_VNIS DEV=$SLINGSHOT_DEVICES'

run() {
  local tag=$1; shift
  echo "################ $tag"
  srun --ntasks-per-node=1 --mpi=none \
    --container-image="$IMG" \
    --container-mounts=/opt/cray:/opt/cray:ro,$HOSTLIBS:/hostlibs:ro,$REPO_DIR/tools:/tools:ro \
    --container-workdir=/tools \
    bash -c "export PATH=/usr/local/bin:\$PATH NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET BENCH_TAG=$tag $*
             slurm2pytorch torchrun --nproc_per_node=4 --nnodes=$SLURM_JOB_NUM_NODES --rdzv_backend=c10d \
               --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_id=$tag nccl_bench.py" \
    > "$R/$tag.log" 2>&1
  grep -h BENCH "$R/$tag.log"
  grep -hoE "Using network [A-Za-z ]+|NET/Socket : Using .*|Selected provider is [a-z]+.*" "$R/$tag.log" | sort -u | head -3
}

run A_default
run B_socket_hsn NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn
run C_ofi_cxi LD_LIBRARY_PATH=$CRAY_LIBFABRIC/lib64:/hostlibs:\$LD_LIBRARY_PATH \
  NCCL_NET_PLUGIN=aws-ofi NCCL_NET=\"AWS Libfabric\" FI_PROVIDER=cxi \
  FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072 NCCL_CROSS_NIC=1
