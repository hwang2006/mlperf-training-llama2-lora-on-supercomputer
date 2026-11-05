# Results

Measured on KISTI Hangang between October 31 and November 6, 2026.
Data: [`results/official_rehearsal.csv`](../results/official_rehearsal.csv),
[`results/exploration_runs.csv`](../results/exploration_runs.csv),
[`results/krai_isambard_baseline.csv`](../results/krai_isambard_baseline.csv).

## 1. Official-submission rehearsal (10 runs per scale, v6.1 package checker)

Score = mean TTQ without the fastest and slowest run x RCP normalization (`scaling.json`
written by the package checker). All seven scales: `PACKAGE CHECKER FOUND NO ERRORS`.

| Nodes | GPUs | TP | CP | GBS | LR / steps | Min TTQ | Olympic mean | RCP norm. | **Score** | Non-conv. |
|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 1 | 8 | 0.00055 / 800 | 53.772 | 62.164 | 1.0347 | **64.323** | 0 |
| 2 | 8 | 4 | 1 | 8 | 0.00055 / 800 | 31.143 | 31.426 | 1.0347 | **32.517** | 0 |
| 4 | 16 | 4 | 1 | 8 | 0.00055 / 800 | 13.798 | 15.701 | 1.0511 | **16.504** | 0 |
| 8 | 32 | 4 | 1 | 8 | 0.00055 / 800 | 8.295 | 8.370 | 1.0347 | **8.660** | 0 |
| 16 | 64 | 4 | 2 a2a | 8 | 0.00055 / 800 | 5.419 | 5.830 | 1.0034 | **5.850** | 1 |
| 32 | 128 | 4 | 2 a2a | 16 | 0.00055 / 800 | 3.196 | 3.734 | 1.0500 | **3.921** | 1 |
| 64 | 256 | 4 | 2 a2a | 32 | 0.0003 / 1024 | 3.070 | 3.434 | 1.0000 | **3.434** | 0 |

Times in minutes. Each run is a separate Slurm job with a distinct seed; job IDs are in the CSV.

### Comparison with KRAI (official MLPerf Training v6.0 scores)

| Nodes | KRAI olympic | KRAI norm. | KRAI score | Hangang score | Hangang vs KRAI | Hangang eff. | KRAI eff. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64.726 | 1.0347 | 66.973 | 64.323 | 4.0% faster | - | - |
| 2 | 31.375 | 1.0681 | 33.512 | 32.517 | 3.0% faster | 98.9% | 99.9% |
| 4 | 17.162 | 1.0034 | 17.220 | 16.504 | 4.2% faster | 97.4% | 97.2% |
| 8 | 9.097 | 1.0000 | 9.097 | 8.660 | 4.8% faster | 92.8% | 92.0% |
| 16 | 5.933 | 1.0000 | 5.933 (GBS 16) | 5.850 (GBS 8) | 1.4% faster | 68.7% | 70.6% |
| 32 | - | - | - | 3.921 | - | 51.3% | - |
| 64 | - | - | - | 3.434 | - | 29.3% | - |

Efficiency = (1-node score / N-node score) / N. KRAI's scores were recomputed from the published
logs with `tools/krai_baseline.py` and the same package checker; they match KRAI's published values.

Interpretation:

* Both systems scale almost identically on official-style scores. Hangang's advantage is a roughly
  constant ~4% per-node speed; both systems have GH200 with 96 GB HBM3, and the most likely cause
  is the power limit (900 W per GH200 on liquid-cooled Hangang, 660 W shared CPU+GPU cap on
  air-cooled Isambard-AI). This was not measured directly.
* Earlier single-run comparisons suggested Hangang scales better than KRAI at 4-16 nodes; with 10
  runs and RCP normalization that difference disappears.

### Runs per scale (TTQ in minutes; `-` = not inspected individually)

| Nodes | TTQ of each run | Notes |
|---:|---|---|
| 1 | 62.017, 62.375, 53.772, 62.378, 70.119, 62.230, 62.291, 62.517, 61.583, 61.922 | one early (2688 samples) and one late (3456) convergence |
| 2 | 31.143, 31.510, 31.534, 31.503, 31.149, 31.291, 31.644, 31.390, 35.259, 31.389 | 35.259: converged at 3456 samples (throughput normal) |
| 4 | 15.890, 15.852, 14.194, 15.920, 13.798, -, -, -, -, - (max 16.397) | two runs at 2688 samples |
| 8 | 8.295, 8.305, 8.308, 8.359, 8.306, -, -, -, -, - (max 9.458) | |
| 16 | NaN, 6.262, 5.419, 6.303, 5.522, 5.501, 5.618, 5.755, 5.554, 6.124 | NaN run used node1000-1007 |
| 32 | 3.636, 3.196, 3.842, 3.270, 3.697, 4.112, 3.593, NaN, 3.823, 3.898 | NaN run used node1000-1007 |
| 64 | 3.475, 3.278, 3.070, 3.261, 3.544, -, -, -, -, - (max 3.979) | 6144-6912 samples in inspected runs |

### Non-converged runs and node1000-1007

Two runs (32 nodes, job 1221790; 16 nodes, job 1221926) produced `NaN` training loss from the
**first** logged step, and every evaluation was `NaN`; both ended `aborted` at max steps. No Xid
or CUDA errors appeared in the job output. The nodes common to both jobs that were never used by
any of the successful jobs (598 distinct nodes) are node1000-1007. A user-level GEMM check
(`tools/gpu_check.py`, FP32/BF16) on these nodes passed, so the fault is not reproducible with
plain GEMMs; the nodes are excluded (`EXCLUDE_NODES`) and should be reported to KISTI for DCGM
diagnostics. Per the rules, each failed run stays in its set and is scored as the slowest run.

## 2. Configuration studies

### 16 nodes (single runs)

| Run | CP | MINIBS | GBS | Samples | Avg. throughput incl. eval | TTQ |
|---|---:|---:|---:|---:|---:|---:|
| Krai config | 2 | 1 | 8 | 3072 | 9.46 | **5.415** |
| E1 | 1 | 1 | 16 | 4224 | 12.38 | 5.780 |
| E2 | 2 | 2 | 16 | 4224 | 10.03 | 7.022 |

Dropping inter-node CP raises throughput by about 21-24%, but GBS 16 needs about 37.5% more
samples at LR 0.00055. Keeping GBS 8 is best at 16 nodes.

### 32 nodes: CP2 `a2a` vs CP4 `p2p` (10 runs each)

| Config | GBS | Train throughput | Converged samples | RCP norm. | Score |
|---|---:|---:|---|---:|---:|
| **CP2 `a2a`** | 16 | ~19.5 samples/s | 3456-4224 (one NaN run) | 1.0500 | **3.921** |
| CP4 `p2p` | 8 | ~15 samples/s | 3072 x4, 3456 x3, 3840 x3 | 1.0000 | 4.319 |

CP4 needs `p2p` (`a2a` fails: "number of attention heads needs to be divisible by CP size",
because TP=4 leaves 2 KV heads per GPU). `p2p` converged later than every other GBS 8
configuration (30 runs at 1-8 nodes mostly at 3072 samples), so it receives no RCP
normalization benefit. Five of the CP4 runs were placed in a single Dragonfly group with
`--switches=1`; throughput did not improve (14.87 vs 15.13 samples/s scattered).

### 64 nodes: GBS, CP and learning rate (single runs)

| Config | GBS | LR / steps | Samples | Train throughput | TTQ | RCP (limit 5802 at GBS 32) |
|---|---:|---|---:|---:|---:|---|
| CP2 `a2a` | 32 | 0.00055 / 800 | 4992 | 40.4 | 2.445 | **fails** (too fast) |
| CP2 `a2a` | 32 | 0.0004 / 1024 | 5376 | 33.0 | 4.005 | fails |
| **CP2 `a2a`** | 32 | **0.0003 / 1024** | 6144 | 36.8 | 3.217 | **passes** |
| CP4 `p2p` | 16 | 0.00055 / 800 | 4224 | 22.6 | 4.278 | passes |

The single-run 2.445 min shown in earlier versions of this repository is not a valid MLPerf
score: its convergence is below the RCP lower limit. Throughput at 64 nodes varied between 33 and
40 samples/s across runs with the same configuration; Slurm's `--switches` (Hangang caps the wait
at 5 minutes) did not reliably keep 64 nodes in one 104-node Dragonfly group.

## 3. NCCL transport (2 nodes, 8 GPUs, all-reduce bus bandwidth)

| Mode | 1 MB | 16 MB | 128 MB | 512 MB | NCCL network |
|---|---:|---:|---:|---:|---|
| A: image as-is | 1.22 | 9.63 | 9.74 | 11.87 GB/s | Socket (`bond0` + `hsn0-3`) |
| B: socket on `hsn*` only | 1.13 | 11.24 | 13.59 | 13.21 GB/s | Socket (`hsn0-3`) |
| C: host Cray libfabric | **5.90** | **74.97** | **156.03** | **178.61 GB/s** | AWS Libfabric, provider `cxi`, 4 NICs |

Despite the ~15x bandwidth gap, 2-node TTQ was 31.557 min (A) vs 31.143 min (C): with TP inside
the node only small LoRA gradients and FP8 amax reductions cross nodes. All multi-node results
above use C.

## 4. Earlier home-built container

| Config | 1 node | 2 nodes |
|---|---:|---:|
| Custom NeMo/Megatron FP8 container, TP overlap off | 72.530 min | 133.213 min (inverse scaling) |
| This repo (NVIDIA v5.0 recipe + KRAI configs), single runs | 61.856 min | 31.143 min |
