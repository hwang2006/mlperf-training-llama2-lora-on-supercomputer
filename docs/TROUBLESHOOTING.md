# Troubleshooting on KISTI Hangang

Every issue below was hit while preparing this benchmark. Symptoms are shown as they
appear in logs.

## Slurm submission

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: Invalid --comment value` / `submission refused by cli_filter` | Hangang requires `--comment="field=<field>;appl=<program>"` | Use values from `showappl`, e.g. `field=efficientai;appl=megatron-lm` |
| `QOSMaxSubmitJobPerUserLimit` | At most 5 jobs per user queued or running (QOS `gpu`) | `scripts/run_series.sh` submits as slots free up |
| `--dependency`/`--begin` jobs also rejected | Pending jobs count toward the 5-job limit | Use `run_series.sh` (started with `setsid nohup`; cron is not allowed) |
| `scontrol update ... job modification is blocked by the field policy` | Pending jobs cannot be edited | Cancel and resubmit |
| `--switches=1@00:30:00` shows `Switches=1@00:05:00` | Site caps the wait for topology placement at 5 min | Expect scattered placement for 64 nodes after 5 min |
| `working directory resolves to /home01/... jobs must run under /scratch or /appsdata` | Jobs may not be submitted from `$HOME` | `cd /scratch/$USER/...` before `sbatch` (`submit.sh` checks this) |
| Background `nohup srun ...` disappears after logout | Login session cleanup | Use `sbatch` for anything longer than a few minutes |

## Container image

| Symptom | Cause | Fix |
|---|---|---|
| `docker://krai4ai/mlperf-nvidia:llama2_70b_lora-pyt ... 401 Unauthorized` | KRAI's image is private/removed | Rebuild from the NVIDIA v5.0 recipe (this repo) |
| `enroot-mksquashovlfs: failed to mount overlay ... Invalid argument` | `ENROOT_TEMP_PATH` on Lustre | Node-local temp (`/tmp/$USER`) |
| `unsquashfs ... FATAL ERROR: Failed to create thread` | Per-user thread limit on login nodes (144 cores) | `export ENROOT_MAX_PROCESSORS=8` |
| `enroot-switchroot: failed to execute: /bin/sh: Permission denied` | `/dev/shm` is mounted `noexec` on login nodes | Use `/tmp/$USER` for `ENROOT_DATA_PATH` |
| After `pip install -e ".[nlp]"`: torch 2.15, `nvidia-*-cu13` installed | Dockerfile sets `PIP_CONSTRAINT=""`; 2026 PyPI resolution pulls newer torch | Pin NGC packages + `uv --exclude-newer 2025-05-01` (see `container/build_inside.sh`) |
| apt `Ign: ports.ubuntu.com` then exit 100 on compute nodes | No Ubuntu mirror access from compute nodes | Build on a login node; `libopencc-dev` failure is tolerated |
| `CUDA initialization: ... out of memory` during build on login node | Login-node GPU state | Harmless for the build |

## Job startup

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Training or validation data not found in /scratch/...` | A host variable named `DATA_ROOT` was exported and Pyxis passed it into the container, overriding the NeMo config's `oc.env:DATA_ROOT` (default `/data`) | Never export `DATA_ROOT`/`CKPT_ROOT` on the host; `run_ttq.sbatch` sets them to `/data` and `/ckpt` |
| `OSError: [Errno 30] Read-only file system: '/root/.cache/huggingface'` | Pyxis rootfs is read-only | `--container-writable`; `HF_HOME`, `XDG_CACHE_HOME`, `HOME` on `/results` |
| `DistNetworkError: client socket has timed out after 60000ms ... (nodeXXXX, 29500)` | torchrun rendezvous default 60 s; master not up | `--rdzv_conf=read_timeout=600,join_timeout=300`; check the master node (next row) |
| `srun: error: task 0 launch failed: Error configuring interconnect` | Slingshot VNI/CXI service setup failed on that node (seen repeatedly on `node0943`) | Exclude the node (`EXCLUDE_NODES` in `env.sh`) and report it to KISTI |
| `torch/_inductor/runtime/triton_heuristics.py ... KeyError: "Unknown key: 'cubin'"`, then no output and no `warmup_time` | 256 ranks sharing one Triton cache directory on Lustre | Node-local `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` (default in `run_ttq.sbatch`) |
| Single-node job exits in <1 min: `NCCL WARN Error: network AWS Libfabric not found` | CXI forced on 1 node; Slurm allocates no Slingshot VNI for single-node jobs | `run_ttq.sbatch` skips the Slingshot settings when N=1 |
| Training loss `NaN` from the first logged step, all evals `NaN`, run `aborted` | Faulty node (seen with node1000-1007; plain GEMM checks pass) | Exclude the nodes (`EXCLUDE_NODES`), report to KISTI; keep the run in its set (scored as slowest) |

## Training configuration

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: The number of attention heads needs to be divisible by CP size!` with CP=4 | Llama2-70B has 8 KV heads -> 2 per rank at TP=4; `a2a` CP needs heads % CP == 0 | Use CP <= 2, or `CP_COMM_TYPE=p2p` (ring attention) for CP 4/8. Note: `p2p` converged later on average and scored worse at 32 nodes |
| Same assertion although `CP_COMM_TYPE=p2p` was passed | Override not exported (check `EXPTAG`/`CP_COMM_TYPE=` lines at the top of the log) | Put `CP_COMM_TYPE` in a config file (`configs/`) instead of an override |
| `ValueError: val_check_interval (288) must be <= number of training batches (2)` | Smoke test with very few steps | Keep validation and MLPerf stopping criteria; do not use 2-step smoke tests |
| Logged GBS differs from expectation | `GBS = floor(MINIBS * floor(GPUs/(TP*PP)) / CP)` | Check `"global_batch_size"` in the log; adjust `MINIBS`/`CP` |
| Override like `CP_COMM_TYPE=p2p` seems ignored | Variable not exported into the job | Check `EXPTAG=...` and `CP_COMM_TYPE=...` lines at the top of the job log |

## MLPerf checkers and packaging

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'dataclasses'` | Default `python3` on login nodes is 3.6 | Use `python3.12 -m pip install --user ...` and `python3.12 -m mlperf_logging...` |
| `pip install --user` succeeds but `python3` cannot import | `pip` belongs to python3.12 | Always `python3.12 -m pip` |
| `OpenBLAS blas_thread_init: pthread_create failed`, then hang | Per-user process limit on login nodes | `OPENBLAS_NUM_THREADS=1` (set by `check_package.sh`) |
| v6.1: `Required EXACTLY_ONE occurrence of 'micro_batch_size'` (and 8 other keys) | NVIDIA v5.0 code does not log the keys made mandatory in v6.1 | `tools/log_disclosure.py` (called by `run_ttq.sbatch`) |
| `Required AT_LEAST_ONE occurrence of 'cache_clear'` | No cache-clear event logged | See docs/OFFICIAL_REHEARSAL.md section 2; logged by `log_disclosure.py` |
| `Unknown files in result directory` | `scaling.json` from a previous checker run | `check_package.sh` deletes it before rerunning (keep the fresh one) |
| `found 7, expected 10` result files | A job was still running when assembling (counted as failed) | `assemble_package.sh` now stops if a job is still in the queue |
| `RCP Test Failed` | Mean converged samples below the RCP lower limit for the GBS | Lower LR / longer schedule for that GBS (e.g. 64 nodes GBS 32: LR 0.0003, 1024 steps) |
| Compliance passes but logs show `"submission_org": "SUBMISSION_ORG_PLACEHOLDER"` | Not checked; KRAI's published logs have the same | Submitter name lives in the directory and system JSON |
| `sudo: a password is required` for `drop_caches` | No sudo for users | ATOM jobEnd flushes caches; run one MLPerf run per job |
| `srun` step end does not flush the page cache | ATOM `appEnd` tasks do not fire for user `srun` steps | Only the job end flushes; one run per job |

## Network (NCCL over Slingshot)

| Symptom | Cause | Fix |
|---|---|---|
| `NET/OFI Couldn't create domain. RC: -38, ERROR: Function not implemented` then `network AWS Libfabric not found` | Container's libfabric lacks/does not match CXI, or `NCCL_NET` forced on a single node (no VNI) | Mount `/opt/cray`, put `$CRAY_LIBFABRIC/lib64` + collected deps first in `LD_LIBRARY_PATH`; do not force `NCCL_NET` on 1 node |
| `Using network Socket` and `NET/Socket : Using [0]bond0 ...` | Image's EFA libfabric has no `cxi` provider | `NETMODE=C` (default) |
| `fi_info: error while loading shared libraries: libcxi.so.1` inside container | Ubuntu container does not search `/usr/lib64` | Copy only the needed libs to `hostlibs/` (done automatically); do not prepend all of host `/usr/lib64` |
| `SLINGSHOT_VNIS=` empty | Single-node step, no VNI allocated | Normal; multi-node steps get e.g. `VNIS=7539` |

To verify the transport in a training run, submit with `NCCL_DEBUG=INFO` and grep for
`Using network AWS Libfabric` and `Selected provider is cxi, fabric is cxi (found 4 nics)`.

## Abandoned path: TP communication overlap

Enabling Transformer Engine userbuffer TP overlap (`TP_COMM_OVERLAP=True`, `UB_SKIPMC=1`)
required MPI-based launching and failed in TE (`AttributeError: 'NoneType' object has no
attribute 'clone'` in `layernorm_linear.py`, FC1 forward all-gather overlap path). KRAI's
submission does not enable TP overlap either (`TP_COMM_OVERLAP` unset -> `False`), so it is
off in this recipe.
