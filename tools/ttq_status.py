#!/usr/bin/env python3
"""Summarize MLPerf logs (:::MLLOG lines) of one or more runs.

Usage:  python3 tools/ttq_status.py results/runs/kv6-ttq-*.out
Prints progress, throughput, evaluation losses and the TTQ (run_start -> run_stop).
"""
import json
import re
import sys

PAT = re.compile(r":::MLLOG (\{.*\})")

for path in sys.argv[1:]:
    run_start = run_stop = None
    gbs = None
    train, evals, thr = [], [], []
    with open(path, errors="ignore") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            try:
                j = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            key, md = j["key"], j.get("metadata", {})
            if key == "global_batch_size":
                gbs = j["value"]
            elif key == "run_start":
                run_start = j["time_ms"]
            elif key == "run_stop":
                run_stop = (j["time_ms"], md.get("status"))
            elif key == "train_loss":
                train.append((j["time_ms"], md.get("samples_count")))
            elif key == "eval_accuracy":
                evals.append((md.get("samples_count"), round(j["value"], 4)))
            elif key == "tracked_stats" and isinstance(j["value"], dict) and "throughput" in j["value"]:
                thr.append(j["value"]["throughput"])

    print(f"## {path}  (GBS={gbs})")
    if run_start and train:
        t, s = train[-1]
        el = (t - run_start) / 1000
        print(f"   {el/60:.1f} min after run_start, samples={s}, avg incl. eval {s/el:.3f} samples/s")
    if thr:
        print(f"   train throughput (excl. eval): {sum(thr)/len(thr):.3f} samples/s")
    if evals:
        print("   EVAL", evals)
    if run_stop and run_start:
        print(f"   RUN_STOP status={run_stop[1]}  TTQ={(run_stop[0]-run_start)/60000:.3f} min")
