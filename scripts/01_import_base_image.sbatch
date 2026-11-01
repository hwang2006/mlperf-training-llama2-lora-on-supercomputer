#!/bin/bash
# Import the NGC PyTorch 25.04 arm64 base image as a squashfs file.
# Submit from the repository root (must be under /scratch):
#   sbatch scripts/01_import_base_image.sbatch
#SBATCH -J kv6-import
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --comment="field=efficientai;appl=megatron-lm"
#SBATCH -o results/kv6-import-%j.log

set -x
source "$SLURM_SUBMIT_DIR/env.sh"
mkdir -p "$CONT_DIR" "$ENROOT_CACHE_PATH"

# enroot creates an overlay mount during squashfs creation; this fails on Lustre
# ("failed to mount overlay ... Invalid argument"), so use node-local temp space.
AVAIL_GB=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
if [ "$AVAIL_GB" -ge 60 ]; then LOCAL=/tmp/$USER; else LOCAL=/dev/shm/$USER; fi
export ENROOT_TEMP_PATH=$LOCAL/enroot-tmp
mkdir -p "$ENROOT_TEMP_PATH"

rm -f "$BASE_IMG"
enroot import -o "$BASE_IMG" "docker://nvcr.io#nvidia/pytorch:25.04-py3"
RC=$?
rm -rf "$ENROOT_TEMP_PATH"
echo "IMPORT_EXIT=$RC"
ls -lh "$BASE_IMG"
