#!/bin/bash
# Per system: (mean TTQ without fastest and slowest run) x (RCP normalization from scaling.json).
# A non-converged run counts as the slowest (it has the largest TTQ or no TTQ at all).
cd "$(dirname "$0")/.."
source ./env.sh
for d in "$SUBMISSION_DIR/$SUBMITTER"/results/Hangang-GH200_n*; do
  s=$(basename "$d"); r=$d/llama2_70b_lora
  nf=$(grep -oE '[0-9]+\.[0-9]+' "$r/scaling.json" 2>/dev/null | head -1); nf=${nf:-1.0}
  for f in "$r"/result_*.txt; do
    t=$(python3 tools/ttq_status.py "$f" | grep -oP 'TTQ=\K[0-9.]+')
    grep -q '"status": "success"' "$f" && echo "$t" || echo inf
  done | sort -g | awk -v s="$s" -v nf="$nf" '{a[NR]=$1} END{t=0; for(i=2;i<NR;i++) t+=a[i]; m=t/(NR-2);
      printf "%-18s runs=%2d  min=%7.3f  olympic=%7.3f  x%.4f  score=%7.3f min\n", s, NR, a[1], m, nf, m*nf}'
done | sort -t_ -k2 -V
