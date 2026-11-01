#!/bin/bash
# Build the benchmark image ($IMG) from the base image ($BASE_IMG) with enroot.
# Run on a Hangang LOGIN node (aarch64, has network access to GitHub/PyPI/Ubuntu mirrors):
#   ./scripts/02_build_container.sh 2>&1 | tee results/kv6-build.log
# Takes ~15-30 min (most of it is `enroot export` compression).
set -euo pipefail
source "$(dirname "$0")/../env.sh"

[ "$(uname -m)" = "aarch64" ] || { echo "Must run on an aarch64 node (got $(uname -m))"; exit 1; }
[ -f "$BASE_IMG" ] || { echo "Base image not found: $BASE_IMG (run 01_import_base_image.sbatch)"; exit 1; }
[ -d "$NV_IMPL" ]  || { echo "NVIDIA sources not found (run 00_fetch_sources.sh)"; exit 1; }

# Node-local, exec-able enroot paths. On Hangang login nodes /dev/shm is noexec
# ("/bin/sh: Permission denied"), so use /tmp/$USER.
export ENROOT_TEMP_PATH=$ENROOT_LOCAL/enroot-tmp
export ENROOT_DATA_PATH=$ENROOT_LOCAL/enroot-data
export ENROOT_RUNTIME_PATH=$ENROOT_LOCAL/enroot-run
# unsquashfs/mksquashfs spawn one thread per core (144); the per-user thread limit
# on login nodes makes that fail with "Failed to create thread".
export ENROOT_MAX_PROCESSORS=8
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH"
df -h /tmp

NAME=kv6-build
enroot remove -f "$NAME" 2>/dev/null || true
enroot create -n "$NAME" "$BASE_IMG"

enroot start --rw --root \
  --mount "$NV_IMPL:/src" \
  --mount "$REPO_DIR/container:/build" \
  "$NAME" bash /build/build_inside.sh

rm -f "$IMG"
enroot export -o "$IMG" "$NAME"
enroot remove -f "$NAME"
ls -lh "$IMG"
echo "BUILD_EXIT=0"
