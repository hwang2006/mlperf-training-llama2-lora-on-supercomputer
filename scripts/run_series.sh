#!/bin/bash
# Submit a series of MLPerf runs (default 10) for one node count, one Slurm job per run,
# filling free slots of the per-user job limit (MAX_JOBS) as earlier jobs finish.
# Job IDs are appended to results/series-n<NODES>.jobs (input for assemble_package.sh).
#
#   ./scripts/run_series.sh NODES SEED_START [COUNT] [extra submit.sh KEY=VALUE ...]
#
# Runs unattended and survives logout (cron is not available on Hangang):
#   setsid nohup ./scripts/run_series.sh 1 1000 > results/series-n1.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source ./env.sh
NODES=${1:?NODES}; START=${2:?SEED_START}; shift 2
COUNT=10
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then COUNT=$1; shift; fi
JOBS=results/series-n${NODES}.jobs
end=$(( $(date +%s) + 12*3600 ))
i=0
while [ $i -lt "$COUNT" ] && [ "$(date +%s)" -lt $end ]; do
  if [ "$(squeue -u "$USER" -h | wc -l)" -lt "$MAX_JOBS" ]; then
    seed=$((START + i))
    out=$(./submit.sh "$NODES" SEED=$seed "$@" 2>&1) && jid=$(echo "$out" | grep -oP 'Submitted batch job \K[0-9]+')
    if [ -n "${jid:-}" ]; then
      echo "$jid" >> "$JOBS"; echo "$(date '+%F %T') n$NODES SEED=$seed job=$jid"; i=$((i+1)); jid=""
    else
      echo "$(date '+%F %T') submit failed: $out"
    fi
  fi
  sleep 60
done
echo "$(date '+%F %T') done: $i of $COUNT submitted; job IDs in $JOBS"
