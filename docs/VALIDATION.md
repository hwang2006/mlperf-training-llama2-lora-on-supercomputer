# Validating a fresh checkout on Hangang

A short test plan to confirm that this repository reproduces the rehearsal, reusing an existing
benchmark image and dataset (set `IMG`, `DATADIR`, `MODEL` in `env.sh`). Run everything from a
**new** clone under `/scratch/$USER`.

| # | Test | Cost | Pass criteria |
|---|---|---|---|
| 0 | `./scripts/00_fetch_sources.sh` | none | `configs/` contains `config_gh200.sh` and `config_common.sh` |
| 1 | `./submit.sh 1 SEED=1100` | ~70 min, 1 node | `EXPTAG=n1-mb8-cp1...`, no `AWS Libfabric not found`, `RUN_STOP status=success`, TTQ ~62 min |
| 2 | `./submit.sh 2 SEED=2100` | ~37 min, 2 nodes | `CACHE_STATE` lines (~0.8 GiB), TTQ ~31 min, `compliance_checker` 6.1.0 `SUCCESS` on `run.log` |
| 3 | `./submit.sh 64 SEED=64100` | ~10 min, 64 nodes | `EXPTAG=n64-mb1-cp2-lr0.0003-ms1024`, `config_filename` = `config_hangang_n64_tp4pp1cp2_lr3e-4.sh`, converged samples >= 5802 |
| 4 | Package from existing logs (copy 10 `run.log` directories into `results/runs/`, list job IDs in a file) | none | `assemble_package.sh` -> `check_package.sh` `SUCCESS` -> `scores.sh` reproduces the published score |
| 5 (optional) | Container build from scratch with a new `IMG` name | ~40 min, login node | `FINAL torch 2.7.0a0...` and `BUILD_EXIT=0` |

Commands for test 2's compliance check:

```bash
f=$(ls -d results/runs/n2-*-<jobid>/run.log)
OPENBLAS_NUM_THREADS=1 python3.12 -m mlperf_logging.compliance_checker --usage training --ruleset 6.1.0 "$f" | tail -3
```

Test 4 can also be checked offline against KRAI's own logs, which the scripts reproduce exactly
(KRAI 2 nodes: olympic 31.375 min x 1.0681 = 33.512 min):

```bash
K=$KRAI_RESULTS/Isambard-AI-Phase2-GH200-96GB_n2_v6.0/llama2_70b_lora
KEYS='lowest_numerical_precision_in_linear|lowest_numerical_precision_in_attn|lowest_numerical_precision_in_comm|tensor_parallelism|pipeline_parallelism|context_parallelism|expert_parallelism|micro_batch_size|config_filename|cache_clear'
j=900000; : > results/series-n2-krai.jobs
for f in $K/result_*.txt; do
  d=results/runs/n2-krai-$j; mkdir -p $d
  { python3 tools/log_disclosure.py 4 1 1 1 config_gh200_n2_tp4pp1cp1.sh; grep -vE "\"key\": \"($KEYS)\"" $f; } > $d/run.log
  echo $j >> results/series-n2-krai.jobs; j=$((j+1))
done
SUBMITTER=TEST ./scripts/assemble_package.sh 2 results/series-n2-krai.jobs
SUBMITTER=TEST ./scripts/check_package.sh && SUBMITTER=TEST ./scripts/scores.sh
rm -rf submission/TEST results/runs/n2-krai-* results/series-n2-krai.jobs
```
