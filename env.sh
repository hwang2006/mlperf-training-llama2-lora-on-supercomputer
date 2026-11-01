# =============================================================================
# env.sh -- central configuration for all scripts in this repository.
# Edit the paths below for your account, then:   source env.sh
#
# KISTI Hangang notes:
#   * Jobs must be submitted from under /scratch or /appsdata (cli_filter).
#   * Every job needs --comment="field=<field>;appl=<program>" (see `showappl`).
# =============================================================================

# Repository root (directory containing this file)
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Containers ---------------------------------------------------------------
export CONT_DIR=/scratch/$USER/containers
export BASE_IMG=$CONT_DIR/kv6-pytorch-25.04-py3-arm64.sqsh      # NGC base image
export IMG=$CONT_DIR/kv6-nv50-llama2lora-arm64.sqsh             # final benchmark image

# --- Dataset and checkpoint (NOT stored in this repo) -------------------------
# DATADIR must contain train.npy and validation.npy
# MODEL   must contain context/ and weights/ (NeMo 2 distributed checkpoint, ~129 GiB)
export DATA_ROOT=/scratch/$USER/mlperf-data/llama2_70b_lora
export DATADIR=$DATA_ROOT/gov_report
export MODEL=$DATA_ROOT/model

# --- Upstream sources (fetched by scripts/00_fetch_sources.sh) ----------------
export THIRD_PARTY=$REPO_DIR/third_party
export NV_IMPL=$THIRD_PARTY/nv50/NVIDIA/benchmarks/llama2_70b_lora/implementations/nyx_n1_ngc25.04_nemo
export KRAI_IMPL=$THIRD_PARTY/krai60/Krai/benchmarks/llama2_70b_lora/implementations
export KRAI_RESULTS=$THIRD_PARTY/krai60/Krai/results

# --- Slingshot (CXI) host libraries ------------------------------------------
export CRAY_LIBFABRIC=/opt/cray/libfabric/2.3.1
export HOSTLIBS=$REPO_DIR/hostlibs          # filled automatically on first run

# --- Outputs ------------------------------------------------------------------
export RESULTS=$REPO_DIR/results/runs

# --- Slurm --------------------------------------------------------------------
export SLURM_PARTITION=normal
export SLURM_COMMENT="field=efficientai;appl=megatron-lm"
export EXCLUDE_NODES=""                     # e.g. "node0943" (nodes with interconnect errors)

# --- Enroot (container build on a login node) ---------------------------------
# Layer cache can live on Lustre; temp/data paths MUST be node-local and exec-able
# (overlayfs fails on Lustre, /dev/shm is mounted noexec on Hangang login nodes).
export ENROOT_CACHE_PATH=/scratch/$USER/enroot-tmp/cache
export ENROOT_LOCAL=/tmp/$USER
