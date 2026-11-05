#!/bin/bash
# Assemble the MLPerf submission tree for one node count from 10 runs.
#   ./scripts/assemble_package.sh NODES JOBID x10
#   ./scripts/assemble_package.sh NODES results/series-n<NODES>.jobs
# Rules applied:
#   * jobs still queued/running -> stop (retry later)
#   * at most one non-converging run (kept; scored as slowest); more -> invalid
#   * all runs must log the same config_filename
set -euo pipefail
cd "$(dirname "$0")/.."
source ./env.sh
NN=${1:?NODES}; shift
if [ $# -eq 1 ] && [ -f "$1" ]; then mapfile -t JIDS < "$1"; else JIDS=("$@"); fi
[ ${#JIDS[@]} -eq 10 ] || { echo "need exactly 10 job IDs, got ${#JIDS[@]}"; exit 1; }

SYS=Hangang-GH200_n$NN
ROOT=$SUBMISSION_DIR/$SUBMITTER
R=$ROOT/results/$SYS/llama2_70b_lora
I=$ROOT/benchmarks/llama2_70b_lora/implementations/$SYS
rm -rf "$R" "$I"; mkdir -p "$R" "$I"

i=0; nfail=0; cfgs=()
for j in "${JIDS[@]}"; do
  if squeue -h -j "$j" 2>/dev/null | grep -q .; then echo "job $j is still running: wait and retry"; exit 1; fi
  log=$(ls -d "$RESULTS"/*-"$j"/run.log 2>/dev/null | head -1)
  [ -f "$log" ] || { echo "run.log not found for job $j"; exit 1; }
  if ! grep -q '"key": "run_stop".*"status": "success"' "$log"; then
    nfail=$((nfail+1)); echo "job $j did not converge (counted as slowest run)"
    [ $nfail -le 1 ] || { echo "more than one non-converging run: result invalid"; exit 1; }
  fi
  cfgs+=("$(grep -m1 -oP '"key": "config_filename", "value": "\K[^"]+' "$log")")
  cp "$log" "$R/result_$i.txt"
  echo "result_$i.txt <- job $j (seed $(grep -m1 -oP '"key": "seed", "value": \K[0-9]+' "$log"))"
  i=$((i+1))
done
CFG=$(printf '%s\n' "${cfgs[@]}" | sort -u)
[ "$(echo "$CFG" | wc -l)" -eq 1 ] || { echo "runs used different configs: $CFG"; exit 1; }

# Implementation directory: the code and the one config actually used, plus how to rebuild.
KN16=$KRAI_IMPL/Isambard-AI-Phase2-GH200-96GB_n16_v6.0
cp "$KN16/train.py" "$KN16/config_common.sh" "$KN16/config_gh200.sh" "$I/"
if [ -f "$CONFIGS/$CFG" ]; then cp "$CONFIGS/$CFG" "$I/"
else cp "$KRAI_IMPL/Isambard-AI-Phase2-GH200-96GB_n${NN}_v6.0/$CFG" "$I/"; fi
mkdir -p "$I/slurm" "$I/tools" "$I/scripts" "$I/container"
cp env.sh submit.sh "$I/"
cp slurm/run_ttq.sbatch "$I/slurm/"
cp tools/log_disclosure.py tools/cache_state.sh "$I/tools/"
cp scripts/00_fetch_sources.sh scripts/01_import_base_image.sbatch scripts/02_build_container.sh scripts/03_prepare_data.sh "$I/scripts/"
cp container/build_inside.sh "$I/container/"
sed "s|@CONFIG@|$CFG|; s|@NODES@|$NN|" docs/submission_README.md > "$I/README.md"

[ -f "$ROOT/systems/$SYS.json" ] || "$PYCHECK" scripts/make_system_json.py "$SUBMITTER" "$NN" >/dev/null
echo "== $SYS: $i results, config $CFG, non-converged $nfail"
