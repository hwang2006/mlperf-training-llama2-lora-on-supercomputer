#!/usr/bin/env python3
"""Generate the README figures from results/*.csv.   python3 tools/make_figures.py"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "..", "results")
out = os.path.join(here, "..", "docs", "figures")
os.makedirs(out, exist_ok=True)

best = {}
for r in csv.DictReader(open(os.path.join(res, "hangang_runs.csv"))):
    if r["best_for_scale"] == "yes":
        best[int(r["nodes"])] = r
krai = {int(r["nodes"]): r for r in csv.DictReader(open(os.path.join(res, "krai_isambard_baseline.csv")))}

nodes = sorted(best)
labels = [f"{n} node{'s' if n > 1 else ''}\n{4*n} GPUs" + ("*" if n == 16 else "") for n in nodes]
h = [float(best[n]["ttq_min"]) for n in nodes]
k = [float(krai[n]["score_min"]) if n in krai else None for n in nodes]

# --- Figure 1: TTQ bars with values ------------------------------------------------
x = np.arange(len(nodes)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
kb = ax.bar(x - w/2, [v or 0 for v in k], w, color="#eb6834", hatch="//", edgecolor="white",
            label="KRAI Isambard-AI (official, 10-run score)")
hb = ax.bar(x + w/2, h, w, color="#2a78d6", label="KISTI Hangang (this work, 1 run)")
for b, v in zip(kb, k):
    ax.text(b.get_x() + b.get_width()/2, (v or 0) + 0.8, f"{v:.3f}" if v else "n/a",
            ha="center", va="bottom", rotation=90, fontsize=9, color="#555" if not v else "black")
for b, v in zip(hb, h):
    ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.3f}", ha="center", va="bottom", rotation=90, fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylabel("Time to quality (minutes, lower is better)")
ax.set_ylim(0, 80)
ax.set_title("MLPerf Training Llama2-70B LoRA on GH200: KRAI vs KISTI Hangang")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0, -0.2, "* 16 nodes: KRAI GBS 16, Hangang GBS 8.  Hangang GBS: 8 (1-16 nodes), 16 (32), 32 (64).",
        transform=ax.transAxes, fontsize=8, color="#555")
fig.tight_layout(); fig.savefig(os.path.join(out, "ttq_comparison.png"), dpi=150); plt.close(fig)

# --- Figure 2: speedup ------------------------------------------------------------
base_h = h[0]
sp_h = [base_h / v for v in h]
samples = [int(best[n]["converged_samples"]) for n in nodes]
sp_thr = [s * smp / samples[0] for s, smp in zip(sp_h, samples)]
kn = [n for n in nodes if n in krai]
sp_k = [float(krai[1]["score_min"]) / float(krai[n]["score_min"]) for n in kn]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(nodes, nodes, "k--", lw=1, label="Ideal linear")
ax.plot(nodes, sp_h, "o-", color="#2a78d6", lw=2, label="Hangang: TTQ speedup")
ax.plot(nodes, sp_thr, "s:", color="#2a78d6", lw=1.5, label="Hangang: throughput speedup (excl. extra samples)")
ax.plot(kn, sp_k, "^-", color="#eb6834", lw=2, label="KRAI Isambard-AI: TTQ speedup")
for n, s in zip(nodes, sp_h):
    ax.annotate(f"{s:.1f}x", (n, s), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#2a78d6")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks(nodes, [str(n) for n in nodes]); ax.set_yticks(nodes, [str(n) for n in nodes])
ax.set_xlabel("Nodes (4 x GH200 each)"); ax.set_ylabel("Speedup vs 1 node")
ax.set_title("Scaling of time to quality")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(out, "scaling.png"), dpi=150); plt.close(fig)
print("figures written to", out)
