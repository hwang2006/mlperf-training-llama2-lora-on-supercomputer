#!/usr/bin/env python3
"""Generate the README figures from results/official_rehearsal.csv and results/krai_isambard_baseline.csv.
   python3 tools/make_figures.py"""
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

han = {int(r["nodes"]): r for r in csv.DictReader(open(os.path.join(res, "official_rehearsal.csv")))}
krai = {int(r["nodes"]): r for r in csv.DictReader(open(os.path.join(res, "krai_isambard_baseline.csv")))}
nodes = sorted(han)
h = [float(han[n]["score_min"]) for n in nodes]
k = [float(krai[n]["official_score_min"]) if n in krai else None for n in nodes]
labels = [f"{n} node{'s' if n > 1 else ''}\n{4*n} GPUs" + ("*" if n == 16 else "") for n in nodes]

# --- Figure 1: official scores --------------------------------------------------------
x = np.arange(len(nodes)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
kb = ax.bar(x - w/2, [v or 0 for v in k], w, color="#eb6834", hatch="//", edgecolor="white",
            label="KRAI Isambard-AI (official MLPerf v6.0 score)")
hb = ax.bar(x + w/2, h, w, color="#2a78d6", label="KISTI Hangang (v6.1 package-checker rehearsal, 10 runs)")
for b, v in zip(kb, k):
    ax.text(b.get_x() + b.get_width()/2, (v or 0) + 0.8, f"{v:.3f}" if v else "n/a",
            ha="center", va="bottom", rotation=90, fontsize=9, color="black" if v else "#777")
for b, v in zip(hb, h):
    ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.3f}", ha="center", va="bottom", rotation=90, fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylabel("Score: olympic mean TTQ x RCP normalization (min, lower is better)")
ax.set_ylim(0, 85)
ax.set_title("MLPerf Training Llama2-70B LoRA on GH200: KRAI official vs KISTI Hangang rehearsal")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0, -0.2, "* 16 nodes: KRAI GBS 16, Hangang GBS 8.  Hangang GBS 8 (1-16 nodes), 16 (32 nodes), 32 (64 nodes, LR 0.0003).",
        transform=ax.transAxes, fontsize=8, color="#555")
fig.tight_layout(); fig.savefig(os.path.join(out, "official_scores.png"), dpi=150); plt.close(fig)

# --- Figure 2: speedup from official scores ---------------------------------------------
sp_h = [h[0] / v for v in h]
kn = [n for n in nodes if n in krai]
sp_k = [float(krai[1]["official_score_min"]) / float(krai[n]["official_score_min"]) for n in kn]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(nodes, nodes, "k--", lw=1, label="Ideal linear")
ax.plot(nodes, sp_h, "o-", color="#2a78d6", lw=2, label="Hangang (rehearsal scores)")
ax.plot(kn, sp_k, "^-", color="#eb6834", lw=2, label="KRAI Isambard-AI (official scores)")
for n, s in zip(nodes, sp_h):
    ax.annotate(f"{s:.1f}x ({100*s/n:.0f}%)", (n, s), textcoords="offset points", xytext=(6, -12),
                fontsize=8, color="#2a78d6")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks(nodes, [str(n) for n in nodes]); ax.set_yticks(nodes, [str(n) for n in nodes])
ax.set_xlabel("Nodes (4 x GH200 each)"); ax.set_ylabel("Speedup vs 1 node (score ratio)")
ax.set_title("Scaling of MLPerf score (time to quality incl. RCP normalization)")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(out, "official_scaling.png"), dpi=150); plt.close(fig)
print("figures written to", out)
