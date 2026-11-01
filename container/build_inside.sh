#!/bin/bash
# Runs INSIDE the writable enroot container (as fake root). Reproduces the RUN steps of
# the NVIDIA MLPerf Training v5.0 Llama2-70B LoRA Dockerfile
# (training_results_v5.0/NVIDIA/benchmarks/llama2_70b_lora/implementations/nyx_n1_ngc25.04_nemo/Dockerfile)
# on top of nvcr.io/nvidia/pytorch:25.04-py3.
#
# Two deviations from a plain `docker build`, both needed to get the SAME result in 2026:
#   1. The Dockerfile sets PIP_CONSTRAINT="" (disables NGC's version pins). Resolving
#      dependencies today would pull torch 2.x from PyPI and replace the NGC torch,
#      breaking Transformer Engine. We pin all NGC GPU packages to the versions in the image.
#   2. All other dependencies are resolved as of 2025-05-01 (uv --exclude-newer), i.e.
#      roughly when NVIDIA built the v5.0 submission image.
#
# Mounts expected:  /src = NVIDIA implementation directory (ADD . /workspace/ft-llm)
set -euxo pipefail

export MAX_JOBS=8            # login nodes limit threads per user
export PIP_CONSTRAINT=""
mkdir -p /workspace/ft-llm && cd /workspace/ft-llm
git config --global user.name a && git config --global user.email a

# --- 1. Pin NGC GPU stack --------------------------------------------------
pip list --format=freeze 2>/dev/null \
  | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton|transformer[-_]engine|numpy|apex|flash[-_]attn|nvidia-|cuda-|cupy|tensorrt|onnx)' \
  > /workspace/ngc-pins.txt
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
pip install uv
UV="uv pip install --system --break-system-packages -c /workspace/ngc-pins.txt --exclude-newer 2025-05-01"

$UV numcodecs==0.13.1

# --- 2. NeMo 25.04-alpha.rc1 ------------------------------------------------
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && git checkout 25.04-alpha.rc1
git rev-parse HEAD > /NEMO_COMMIT_HASH.env
$UV -e ".[nlp]"
cd /workspace/ft-llm

# --- 3. Megatron-core 25.04-alpha.rc1 --------------------------------------
uv pip uninstall --system --break-system-packages megatron-core || true
git clone https://github.com/NVIDIA/Megatron-LM Megatron-LM
cd Megatron-LM && git checkout 25.04-alpha.rc1
git rev-parse HEAD > /MCORE_COMMIT_HASH.env
$UV --no-build-isolation .
cd megatron/core/datasets && make && cd /workspace/ft-llm

# --- 4. Transformer Engine: TE_REVISION=SKIP -> keep the TE shipped in the base image

# --- 5. opencc (not on the LoRA training path; tolerate a missing apt mirror) ---
(apt-get update -qq && apt-get install -y -qq --no-install-recommends libopencc-dev) \
  || echo "WARN: libopencc-dev not installed"

# --- 6. ADD . /workspace/ft-llm  +  requirements.txt ------------------------
cp -a /src/. /workspace/ft-llm/
$UV -r requirements.txt

# --- 7. ENV PYTHONPATH (enroot reads /etc/environment) ----------------------
grep -q 'ft-llm/Megatron-LM' /etc/environment || \
  echo 'PYTHONPATH=/workspace/ft-llm/Megatron-LM' >> /etc/environment

# --- 8. Verify -------------------------------------------------------------
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || { echo "ERROR: torch changed $TORCH_BEFORE -> $TORCH_AFTER"; exit 1; }
python - <<'EOF' 2>&1 | grep -vE 'UserWarning|getDeviceCount'
import torch, transformer_engine as te, nemo, megatron.core as mc, mlperf_logging, mlperf_common
print("FINAL torch", torch.__version__, "cuda", torch.version.cuda, "| TE", te.__version__,
      "| NeMo", nemo.__version__, "| MCore", mc.__version__, "| NCCL", torch.cuda.nccl.version())
EOF
which slurm2pytorch

# --- 9. Record and clean -----------------------------------------------------
pip list --format=freeze 2>/dev/null > /workspace/kv6-pip-freeze.txt
echo "NEMO=$(cat /NEMO_COMMIT_HASH.env) MCORE=$(cat /MCORE_COMMIT_HASH.env)" > /workspace/kv6-build-info.txt
rm -rf /root/.cache/pip /root/.cache/uv
apt-get clean || true
echo "BUILD_INSIDE_OK"
