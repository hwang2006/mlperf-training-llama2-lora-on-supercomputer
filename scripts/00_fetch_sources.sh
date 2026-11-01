#!/bin/bash
# Fetch the upstream code this repo builds on (sparse checkouts, a few MB):
#   1. NVIDIA MLPerf Training v5.0 Llama2-70B LoRA implementation (container recipe, configs)
#   2. KRAI MLPerf Training v6.0 Isambard-AI GH200 submission (train.py, configs, result logs)
# Run on a login node from the repository root:   ./scripts/00_fetch_sources.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p "$THIRD_PARTY" && cd "$THIRD_PARTY"

if [ ! -d nv50 ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/mlcommons/training_results_v5.0 nv50
fi
( cd nv50 && git sparse-checkout set --no-cone \
    "/NVIDIA/benchmarks/llama2_70b_lora/implementations/nyx_n1_ngc25.04_nemo/" )

if [ ! -d krai60 ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/mlcommons/training_results_v6.0 krai60
fi
( cd krai60 && git sparse-checkout set --no-cone \
    "/Krai/benchmarks/llama2_70b_lora/implementations/" \
    "/Krai/results/Isambard-AI-Phase2-GH200-96GB_n1_v6.0/" \
    "/Krai/results/Isambard-AI-Phase2-GH200-96GB_n2_v6.0/" \
    "/Krai/results/Isambard-AI-Phase2-GH200-96GB_n4_v6.0/" \
    "/Krai/results/Isambard-AI-Phase2-GH200-96GB_n8_v6.0/" \
    "/Krai/results/Isambard-AI-Phase2-GH200-96GB_n16_v6.0/" )

echo "== NVIDIA v5.0 implementation";  ls "$NV_IMPL"
echo "== KRAI v6.0 implementations";   ls "$KRAI_IMPL" | grep Isambard
