# Changelog

## 2026-11-06 (fix found in validation)

* `env.sh`: host data root renamed `DATA_ROOT` -> `DATA_BASE`. The NeMo config reads `DATA_ROOT`
  inside the container, and Pyxis passed the host value through, so training could not find the
  data. `run_ttq.sbatch` now sets `DATA_ROOT=/data CKPT_ROOT=/ckpt` explicitly.
* `env.sh`: `SLURM_PARTITION`/`SLURM_COMMENT` renamed `JOB_PARTITION`/`JOB_COMMENT` (`SLURM_*`
  variables are also read by `srun`).

## 2026-11-06: official-submission rehearsal

* All seven scales (1-64 nodes) measured with 10 runs each and validated with the MLPerf
  Training v6.1 package checker; scores in `results/official_rehearsal.csv`.
* New: `tools/log_disclosure.py` (v6.1 mandatory keys + `cache_clear`), `tools/cache_state.sh`,
  `tools/gpu_check.py`, `scripts/run_series.sh`, `scripts/make_system_json.py`,
  `scripts/assemble_package.sh`, `scripts/check_package.sh`, `scripts/scores.sh`,
  `configs/` (32/64-node configs), `docs/OFFICIAL_REHEARSAL.md`, `docs/VALIDATION.md`,
  `docs/submission_README.md`.
* `slurm/run_ttq.sbatch`: config files instead of overrides (`CFG=`), disclosure and cache-state
  logging, no Slingshot settings on single-node runs (previously failed with
  "network AWS Libfabric not found").
* `submit.sh`: default configs for 32/64 nodes, `CFG`, `SWITCHES` keys, per-scale wall times.
* `env.sh`: excluded nodes (node0943, node1000-1007), `SUBMITTER`, `PYCHECK`, `MAX_JOBS`.
* Corrections: both Hangang and Isambard-AI GH200 have 96 GB HBM3 ("120GB" is the LPDDR5X per
  Grace); KRAI's official scores include RCP normalization (66.973, 33.512, 17.220, 9.097,
  5.933 min); the 64-node single-run 2.445 min (LR 0.00055) fails the RCP check and is not a
  valid score; the 64-node configuration is now GBS 32 with LR 0.0003 and 1024 steps.

## 2026-11-01: initial version

* NVIDIA v5.0 recipe rebuilt with Enroot, KRAI v6.0 configs, single runs on 1-64 nodes.
