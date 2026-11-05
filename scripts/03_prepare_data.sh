#!/bin/bash
# Download the GovReport dataset (preprocessed for MLPerf) and the Llama2-70B NeMo
# checkpoint using NVIDIA's scripts inside the benchmark image.
# Run on a login node (network access required):
#   ./scripts/03_prepare_data.sh 2>&1 | tee results/kv6-data.log
#
# Result (~130 GiB total):
#   $DATADIR/train.npy, $DATADIR/validation.npy
#   $MODEL/context/{io.json,model.yaml,nemo_tokenizer/}, $MODEL/weights/...
#
# Sources (Hugging Face):
#   dataset    regisss/scrolls_gov_report_preprocessed_mlperf_2
#   checkpoint michal1270/llama2_70b_nemo_zarr
set -euo pipefail
source "$(dirname "$0")/../env.sh"
[ -f "$IMG" ] || { echo "Image not found: $IMG (run 02_build_container.sh)"; exit 1; }

export ENROOT_TEMP_PATH=$ENROOT_LOCAL/enroot-tmp
export ENROOT_DATA_PATH=$ENROOT_LOCAL/enroot-data
export ENROOT_RUNTIME_PATH=$ENROOT_LOCAL/enroot-run
export ENROOT_MAX_PROCESSORS=8
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$DATADIR" "$MODEL" "$DATA_BASE/.hf"

NAME=kv6-data
enroot remove -f "$NAME" 2>/dev/null || true
enroot create -n "$NAME" "$IMG"
enroot start \
  --mount "$DATADIR:/data/gov_report" \
  --mount "$MODEL:/data/model" \
  --mount "$DATA_BASE/.hf:/hf" \
  "$NAME" bash -c '
    set -ex
    export HF_HOME=/hf
    cd /workspace/ft-llm
    python scripts/download_dataset.py --data_dir /data/gov_report
    python scripts/download_model.py   --model_dir /data/model --max_workers 16
  '
enroot remove -f "$NAME"

echo "== dataset";    ls -la "$DATADIR"
echo "== checkpoint"; ls "$MODEL" "$MODEL/context"; du -sh "$MODEL"
