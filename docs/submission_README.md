# Llama2-70B LoRA on Hangang (HPE Cray EX254n, NVIDIA GH200), @NODES@ node(s)

This implementation reuses the NVIDIA MLPerf Training v5.0 Llama2-70B LoRA code
(`training_results_v5.0/NVIDIA/benchmarks/llama2_70b_lora/implementations/nyx_n1_ngc25.04_nemo`)
and Krai's v6.0 Isambard-AI `train.py` and configs (unmodified). Only the launcher, the
configuration for 32/64 nodes and system-specific settings differ.

Config used for these results: `@CONFIG@`

## Files

| File | Purpose |
|---|---|
| `train.py`, `config_common.sh`, `config_gh200.sh` | Krai v6.0 training script and shared configs (unmodified) |
| `@CONFIG@` | Node-count specific config used for these results |
| `slurm/run_ttq.sbatch`, `submit.sh` | Slurm + Pyxis launcher; one MLPerf run per Slurm job |
| `tools/log_disclosure.py` | Logs the v6.1 mandatory disclosure keys and `cache_clear` |
| `tools/cache_state.sh` | Logs per-node page-cache size right before training (`CACHE_STATE` lines) |
| `env.sh` | Paths and Slurm settings (edit first) |
| `scripts/00_fetch_sources.sh` | Fetches the NVIDIA v5.0 and Krai v6.0 sources from MLCommons |
| `scripts/01_import_base_image.sbatch` | Imports `nvcr.io/nvidia/pytorch:25.04-py3` (arm64) with Enroot |
| `scripts/02_build_container.sh`, `container/build_inside.sh` | Rebuild the NVIDIA v5.0 Dockerfile with Enroot |
| `scripts/03_prepare_data.sh` | Downloads the dataset and checkpoint with NVIDIA's scripts |

## 1. Container

Edit `env.sh`, then from this directory:

    ./scripts/00_fetch_sources.sh
    sbatch scripts/01_import_base_image.sbatch     # -> base image .sqsh
    ./scripts/02_build_container.sh                # aarch64 login node -> kv6-nv50-llama2lora-arm64.sqsh

`container/build_inside.sh` runs the `RUN` steps of the NVIDIA v5.0 Dockerfile (base
`pytorch:25.04-py3`, NeMo and Megatron-core `25.04-alpha.rc1`, Transformer Engine from the base
image). The Dockerfile sets `PIP_CONSTRAINT=""`, so a rebuild today would replace the NGC PyTorch;
the script pins all NGC GPU packages, resolves other dependencies as of 2025-05-01
(`uv --exclude-newer`) and aborts if the PyTorch version changes.

## 2. Data

    ./scripts/03_prepare_data.sh
    # dataset:    regisss/scrolls_gov_report_preprocessed_mlperf_2 -> train.npy, validation.npy
    # checkpoint: michal1270/llama2_70b_nemo_zarr -> NeMo 2 distributed checkpoint (context/, weights/)

## 3. Run

    ./submit.sh @NODES@ SEED=<seed>

| Nodes | GPUs | TP | CP | CP comm | MINIBS | GBS | LR / max steps | Config |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 1 | 4 | 4 | 1 | - | 8 | 8 | 0.00055 / 800 | `config_gh200_n1_tp4pp1cp1.sh` (Krai) |
| 2 | 8 | 4 | 1 | - | 4 | 8 | 0.00055 / 800 | `config_gh200_n2_tp4pp1cp1.sh` (Krai) |
| 4 | 16 | 4 | 1 | - | 2 | 8 | 0.00055 / 800 | `config_gh200_n4_tp4pp1cp1.sh` (Krai) |
| 8 | 32 | 4 | 1 | - | 1 | 8 | 0.00055 / 800 | `config_gh200_n8_tp4pp1cp1.sh` (Krai) |
| 16 | 64 | 4 | 2 | a2a | 1 | 8 | 0.00055 / 800 | `config_gh200_n16_tp4pp1cp1.sh` (Krai) |
| 32 | 128 | 4 | 2 | a2a | 1 | 16 | 0.00055 / 800 | `config_hangang_n32_tp4pp1cp2.sh` |
| 64 | 256 | 4 | 2 | a2a | 1 | 32 | 0.0003 / 1024 | `config_hangang_n64_tp4pp1cp2_lr3e-4.sh` |

FP8, CUDA graphs, TP communication overlap off.

## System-specific settings

* **NCCL over Slingshot (multi-node):** the image's EFA libfabric has no CXI provider. The host
  HPE Cray libfabric 2.3.1 (`/opt/cray`) and its dependencies are mounted and put first in
  `LD_LIBRARY_PATH`, with `NCCL_NET_PLUGIN=aws-ofi`, `FI_PROVIDER=cxi`. Single-node runs do not
  use these settings (no inter-node traffic; Slurm allocates no Slingshot VNI for 1 node).
* **Cache clearing:** each MLPerf run is a separate Slurm job. HPE ATOM `jobEnd` tasks on every
  node run `sync && echo 3 > /proc/sys/vm/drop_caches` (`030_clear_vm_cache`) and
  `lctl set_param -t ldlm.drop_caches=1` (`020_clear_lustre_caches_job`), so every run starts on
  nodes with flushed page and Lustre caches. Verified by filling ~6 GiB of page cache in one job
  and measuring 0.8 GiB at the start of the next job on the same nodes. Each result log contains
  the per-node `Cached` size (`CACHE_STATE` lines) measured right before training.
* **Read-only container rootfs:** `--container-writable`; `HF_HOME`, `XDG_CACHE_HOME` and `HOME`
  point to the results mount.
* **Compile caches:** `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` are node-local (`/tmp`).
* **Rendezvous:** `--rdzv_conf=read_timeout=600,join_timeout=300`.
