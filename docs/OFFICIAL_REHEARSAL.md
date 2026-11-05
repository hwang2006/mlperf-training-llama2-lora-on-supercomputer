# Official-submission rehearsal

How to produce an MLPerf Training Llama2-70B LoRA submission package on Hangang and validate it
with the MLPerf checkers. Written against MLPerf Training **v6.1** rules and
`mlperf-logging` ruleset `6.1.0` (the newest available in November 2026). Re-run the checkers
with the next ruleset (for example `7.0.0`) once MLCommons publishes it.

## 1. Rules that shape the procedure

| Rule (training_rules, v6.1) | What this repo does |
|---|---|
| 10 runs per system for "LLM finetuning"; score = mean without fastest and slowest | `run_series.sh` submits 10 runs; `scores.sh` computes the olympic mean |
| One non-converging run may be treated as the slowest; more than one invalidates the result | `assemble_package.sh` keeps one and refuses a second |
| Running several sets and keeping the best is prohibited (M > N runs only with a predefined consecutive-window rule) | Keep every run you start for a set; do not replace a failed run with a new one |
| Flush caches or restart before benchmarking; data must start on durable storage | One run per Slurm job; HPE ATOM `jobEnd` flushes page cache and Lustre cache on every node (verified, see below) |
| RCP: convergence must not be significantly faster than the reference for the GBS; within the band the score is scaled by `RCP mean / submission mean` | `check_package.sh` runs the RCP checker; `scaling.json` holds the factor and stays in the package |
| v6.1: `lowest_numerical_precision_in_{linear,attn,comm}`, `{tensor,pipeline,context,expert}_parallelism`, `micro_batch_size`, `config_filename` must be logged exactly once | `tools/log_disclosure.py` writes them into every result log |
| Hyperparameters: LoRA rank 16, alpha 32, dropout 0.1, grad clip 0.3, weight decay 1e-4, seq 8192 fixed; GBS, LR, steps, warmup free | Krai configs for 1-16 nodes; `configs/` for 32/64 |

RCP limits for Llama2-70B LoRA (`rcps_llama2_70b_lora.json`, ruleset 6.1.0, from the checker):

| GBS | RCP mean (samples) | Lower limit (faster = fail) | Max speedup |
|---:|---:|---:|---:|
| 8 | 3178.7 | 2960.6 | 1.0736 |
| 16 | 4032.0 | 3733.0 | 1.0801 |
| 32 | 6037.3 | 5802.1 | 1.0405 |
| 128 | 11264.0 | 10649.8 | 1.0577 |

The checker compares the mean converged samples of the 10 runs after dropping the fastest and
slowest. At GBS 8 with mostly 3072-sample runs the factor is 3178.7/3072 = 1.0347. If four or
more of ten GBS 8 runs converge at 2688 samples (and the rest at 3072), the set falls below the
lower limit and fails.

## 2. Cache flushing on Hangang

Users have no sudo (`sudo -n sysctl vm.drop_caches=3` asks for a password). Slurm's prolog and
epilog run HPE ATOM (`/opt/cray/atom/tasks/atom_slurm.py`), configured in `/etc/atom.d/`:

| ATOM task | Command | Events |
|---|---|---|
| `030_clear_vm_cache` | `sync && echo 3 > /proc/sys/vm/drop_caches` | appEnd, jobEnd |
| `020_clear_lustre_caches_job` | `lctl set_param -t ldlm.drop_caches=1` | jobEnd |
| `020_clear_lustre_caches` | same | appEnd |

Measured behavior:

* Ending an `srun` step inside a job did **not** flush the page cache (6.1 GiB stayed cached 15 s later).
* Ending a job **did**: job A left 6.1 GiB cached on two nodes; job B on the same nodes started with 0.8 GiB.

Therefore each MLPerf run is its own Slurm job. Every result log records the per-node `Cached`
size right before training (`CACHE_STATE` lines) and a `cache_clear` event whose metadata names
the mechanism. KRAI's v6.0 logs contain a `cache_clear` line without an actual flush (no sudo on
Isambard-AI); do not copy that. Confirm with the MLCommons Training WG that the jobEnd mechanism
is acceptable before an official submission.

## 3. Procedure

### 3.1 Prerequisites

```bash
python3.12 -m pip install --user "git+https://github.com/mlcommons/logging.git"
python3.12 -c "import mlperf_logging, os; print(sorted(d for d in os.listdir(os.path.dirname(mlperf_logging.__file__)+'/compliance_checker') if d.startswith('training_6')))"
```

The default `python3` on Hangang login nodes is 3.6 and cannot run the checkers. `pip` may belong
to another interpreter; always use `python3.12 -m pip`.

### 3.2 Ten runs per scale

```bash
setsid nohup ./scripts/run_series.sh NODES SEED_START > results/series-nNODES.log 2>&1 < /dev/null &
```

* One job per run, distinct seeds (the series uses `SEED_START .. SEED_START+9`).
* Up to 5 jobs are kept in the queue; new ones are submitted as earlier ones finish. The runner
  survives logout (cron is not available) and writes job IDs to `results/series-nNODES.jobs`.
* Runs in parallel on different nodes are allowed; each starts on cache-flushed nodes.
* Approximate cost per set: 1 node ~12 node-h, 2: ~12, 4: ~14, 8: ~19, 16: ~33, 32: ~64, 64: ~139.

Watch the first training loss of each run; `NaN` at the first record indicates a bad node:

```bash
for f in results/runs/kv6-ttq-*.out; do echo "$f $(grep -m1 '"key": "train_loss"' $f | grep -oP '"value": \K[^,]+')"; done
```

### 3.3 Package, check, score

```bash
python3.12 scripts/make_system_json.py            # once; uses SUBMITTER from env.sh
./scripts/assemble_package.sh NODES results/series-nNODES.jobs
./scripts/check_package.sh                         # all scales assembled so far
./scripts/scores.sh
```

Package layout (same as KRAI's v6.0 submission):

```
submission/<SUBMITTER>/
├── systems/Hangang-GH200_n<N>.json
├── results/Hangang-GH200_n<N>/llama2_70b_lora/result_0..9.txt, scaling.json
└── benchmarks/llama2_70b_lora/implementations/Hangang-GH200_n<N>/
    ├── README.md, train.py, config_common.sh, config_gh200.sh, <config used>
    ├── env.sh, submit.sh, slurm/run_ttq.sbatch, tools/{log_disclosure.py,cache_state.sh}
    └── scripts/0[0-3]_*, container/build_inside.sh
```

`check_package.sh` deletes old `scaling.json` files before running so reruns do not warn
("Unknown files in result directory"); the checker writes fresh ones, which belong in the package.

### 3.4 Before an actual submission

| Item | Notes |
|---|---|
| MLCommons membership or test-partner agreement | Required to submit; joint submission with a member is an option |
| Submitter name | Set `SUBMITTER` in `env.sh`, rerun `make_system_json.py` and `assemble_package.sh` (the name appears only in the directory and the system JSON; the logs' `submission_org` field stays a placeholder, as in KRAI's v6.0 logs) |
| Register 8 weeks before the deadline, sign the CLA | Per MLCommons submission rules |
| Ruleset of the round | Rerun `check_package.sh 7.0.0` (or newer) when available; check that Llama2-70B LoRA is still in the suite |
| Cache-clear method | Confirm the ATOM jobEnd approach with the WG |
| Node health | Report node0943 and node1000-1007 to KISTI; consider a reservation for 32/64-node sets |
| 64-node LR | LR 0.0003 passes the RCP check but converges slightly later than the RCP mean; a value between 0.0003 and 0.0004 may lower the score while staying above the lower limit (needs its own 10-run set) |
