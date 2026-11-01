#!/usr/bin/env python3
"""Recompute KRAI's official MLPerf Training v6.0 Llama2-70B LoRA scores on Isambard-AI
from the published result logs (fetched by scripts/00_fetch_sources.sh).

MLPerf score = mean TTQ of the 10 runs after dropping the fastest and the slowest.

Usage:  source env.sh && python3 tools/krai_baseline.py "$KRAI_RESULTS"
"""
import glob
import json
import re
import statistics as st
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "third_party/krai60/Krai/results"
PAT = re.compile(r":::MLLOG (\{.*\})")

print(f"{'nodes':>5} {'GPUs':>4} {'GBS':>4} {'CP':>3} {'score(min)':>10} {'thr(s/s)':>9}  converged samples")
for n in [1, 2, 4, 8, 16]:
    files = sorted(glob.glob(f"{root}/Isambard-AI-Phase2-GH200-96GB_n{n}_v6.0/llama2_70b_lora/result_*.txt"))
    if not files:
        continue
    ttq, samples, thr = [], [], []
    gbs = cp = None
    for fn in files:
        start = stop = last = None
        for line in open(fn, errors="ignore"):
            m = PAT.search(line)
            if not m:
                continue
            j = json.loads(m.group(1))
            k = j["key"]
            if k == "global_batch_size":
                gbs = j["value"]
            elif k == "context_parallelism":
                cp = j["value"]
            elif k == "run_start":
                start = j["time_ms"]
            elif k == "run_stop":
                stop = j["time_ms"]
            elif k == "eval_accuracy":
                last = j["metadata"]["samples_count"]
            elif k == "tracked_stats" and isinstance(j["value"], dict) and "throughput" in j["value"]:
                thr.append(j["value"]["throughput"])
        ttq.append((stop - start) / 60000)
        samples.append(last)
    s = sorted(ttq)
    score = st.mean(s[1:-1]) if len(s) > 2 else st.mean(s)
    print(f"{n:>5} {4*n:>4} {gbs:>4} {cp:>3} {score:>10.3f} {st.mean(thr):>9.3f}  {sorted(set(samples))}")
