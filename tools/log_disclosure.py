#!/usr/bin/env python3
"""Print the MLPerf Training v6.1 mandatory disclosure keys and cache_clear as :::MLLOG lines.

Usage: log_disclosure.py TP PP CP MBS CONFIG_FILENAME [linear attn comm]
Runs on the batch host (any python3 >= 3.6) right before the training step; its output goes
into the same result log as the training output.
"""
import json
import sys
import time

tp, pp, cp, mbs, cfg = sys.argv[1:6]
lin, attn, comm = (sys.argv[6:9] + ["fp8", "fp8", "fp8"])[:3]
t = int(time.time() * 1000)


def log(key, value, lineno, **extra):
    md = {"file": "log_disclosure.py", "lineno": lineno}
    md.update(extra)
    print(":::MLLOG " + json.dumps({"namespace": "", "time_ms": t, "event_type": "POINT_IN_TIME",
                                    "key": key, "value": value, "metadata": md}), flush=True)


items = [("lowest_numerical_precision_in_linear", lin), ("lowest_numerical_precision_in_attn", attn),
         ("lowest_numerical_precision_in_comm", comm), ("tensor_parallelism", int(tp)),
         ("pipeline_parallelism", int(pp)), ("context_parallelism", int(cp)), ("expert_parallelism", 1),
         ("micro_batch_size", int(mbs)), ("config_filename", cfg)]
for i, (k, v) in enumerate(items):
    log(k, v, i + 1)

# Cache flush: each MLPerf run is a separate Slurm job. HPE ATOM jobEnd tasks on every node run
#   030_clear_vm_cache:          sync && echo 3 > /proc/sys/vm/drop_caches
#   020_clear_lustre_caches_job: lctl set_param -t ldlm.drop_caches=1
# so every run starts on nodes with flushed page and Lustre caches (verified on Hangang;
# per-node page-cache size is logged as CACHE_STATE lines by tools/cache_state.sh).
log("cache_clear", True, 10,
    method="HPE ATOM jobEnd drop_caches=3 + lustre ldlm.drop_caches; one run per Slurm job")
