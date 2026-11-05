# =============================================================================
# env.sh -- central configuration for all scripts in this repository.
# Edit the values marked EDIT, then:   source env.sh
#
# KISTI Hangang notes:
#   * Jobs must be submitted from under /scratch or /appsdata (cli_filter).
#   * Every job needs --comment="field=<field>;appl=<program>" (see `showappl`).
#   * At most 5 jobs per user may be queued or running (QOS gpu: MaxSubmitPU=5).
# =============================================================================

# Repository root (directory containing this file)
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Containers (EDIT) --------------------------------------------------------
export CONT_DIR=/scratch/$USER/containers
export BASE_IMG=$CONT_DIR/kv6-pytorch-25.04-py3-arm64.sqsh      # NGC base image
export IMG=$CONT_DIR/kv6-nv50-llama2lora-arm64.sqsh             # benchmark image

# --- Dataset and checkpoint, NOT stored in this repo (EDIT) -------------------
# DATADIR must contain train.npy and validation.npy
# MODEL   must contain context/ and weights/ (NeMo 2 distributed checkpoint, ~129 GiB)
# Do not name a variable here DATA_ROOT or CKPT_ROOT: the NeMo config reads those inside the container.
export DATA_BASE=/scratch/$USER/mlperf-data/llama2_70b_lora
export DATADIR=$DATA_BASE/gov_report
export MODEL=$DATA_BASE/model

# --- Upstream sources (fetched by scripts/00_fetch_sources.sh) ----------------
export THIRD_PARTY=$REPO_DIR/third_party
export NV_IMPL=$THIRD_PARTY/nv50/NVIDIA/benchmarks/llama2_70b_lora/implementations/nyx_n1_ngc25.04_nemo
export KRAI_IMPL=$THIRD_PARTY/krai60/Krai/benchmarks/llama2_70b_lora/implementations
export KRAI_RESULTS=$THIRD_PARTY/krai60/Krai/results
export CONFIGS=$REPO_DIR/configs              # Hangang configs for 32 and 64 nodes

# --- Slingshot (CXI) host libraries ------------------------------------------
export CRAY_LIBFABRIC=/opt/cray/libfabric/2.3.1
export HOSTLIBS=$REPO_DIR/hostlibs            # filled automatically on first multi-node run

# --- Outputs ------------------------------------------------------------------
export RESULTS=$REPO_DIR/results/runs
export SUBMITTER=${SUBMITTER:-KISTI}           # EDIT once the MLCommons submitter is decided
export SUBMISSION_DIR=$REPO_DIR/submission

# --- Slurm (EDIT if needed) ---------------------------------------------------
export JOB_PARTITION=normal
export JOB_COMMENT="field=efficientai;appl=megatron-lm"
export MAX_JOBS=5
# Nodes excluded from all runs:
#   node0943          "Error configuring interconnect" (Slingshot VNI setup fails)
#   node[1000-1007]   two runs that used these nodes produced NaN from the first step
export EXCLUDE_NODES="node0943,node[1000-1007]"

# --- MLPerf checkers (login node) ---------------------------------------------
# The default python3 on Hangang login nodes is 3.6 (no dataclasses); use 3.12.
# (scripts/check_package.sh also limits OpenBLAS threads for the login-node limits)
export PYCHECK=python3.12

# --- Enroot (container build on a login node) ---------------------------------
# Layer cache can live on Lustre; temp/data paths MUST be node-local and exec-able
# (overlayfs fails on Lustre; /dev/shm is mounted noexec on Hangang login nodes).
export ENROOT_CACHE_PATH=/scratch/$USER/enroot-tmp/cache
export ENROOT_LOCAL=/tmp/$USER
