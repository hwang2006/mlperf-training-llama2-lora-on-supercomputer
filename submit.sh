#!/bin/bash
# Submit one MLPerf run (one Slurm job = one run, so every run starts on cache-flushed nodes).
#   ./submit.sh NODES [KEY=VALUE ...]
# Keys:
#   SEED=<int>            random seed (use a distinct seed per run)
#   CFG=<file>            config in configs/ (default for 32/64 nodes: the official-rehearsal config)
#   NETMODE=C|A           C = Slingshot CXI (default), A = TCP sockets
#   TIME=HH:MM:SS         wall time (default by node count)
#   SWITCHES=<n@HH:MM:SS> ask Slurm to place nodes in <= n Dragonfly groups (Hangang caps the wait at 5 min)
#   NCCL_DEBUG=INFO|WARN
#   MINIBS CP LR MAX_STEPS CP_COMM_TYPE   experiment overrides (not for official runs)
# Examples:
#   ./submit.sh 1 SEED=1000
#   ./submit.sh 64 SEED=64000
#   ./submit.sh 32 SEED=32900 CFG=config_hangang_n32_tp4pp1cp4.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
NODES=${1:?usage: ./submit.sh NODES [KEY=VALUE ...]}; shift
case "$PWD" in /scratch/*|/appsdata/*) ;; *) echo "Submit from /scratch or /appsdata (Hangang policy)"; exit 1;; esac

case "$NODES" in
  1) TIME=01:30:00 ;; 2) TIME=01:00:00 ;; 4) TIME=00:45:00 ;; *) TIME=00:40:00 ;;
esac
case "$NODES" in
  32) CFG=config_hangang_n32_tp4pp1cp2.sh ;;
  64) CFG=config_hangang_n64_tp4pp1cp2_lr3e-4.sh ;;
  *)  CFG="" ;;
esac
EXPORTS="ALL"; SW=()
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  case "$k" in
    MINIBS|CP|LR|MAX_STEPS|CP_COMM_TYPE) EXPORTS="$EXPORTS,OV_$k=$v"; echo "WARNING: override $k is not reflected in config_filename" ;;
    NETMODE|SEED|NCCL_DEBUG)             EXPORTS="$EXPORTS,$k=$v" ;;
    CFG)      CFG=$v ;;
    TIME)     TIME=$v ;;
    SWITCHES) SW=(--switches="$v") ;;
    *) echo "Unknown key: $k"; exit 1 ;;
  esac
done
[ -n "$CFG" ] && { [ -f "$CONFIGS/$CFG" ] || { echo "No such config: $CONFIGS/$CFG"; exit 1; }; EXPORTS="$EXPORTS,CFG=$CFG"; }

mkdir -p "$RESULTS"
EXCL=(); [ -n "$EXCLUDE_NODES" ] && EXCL=(--exclude="$EXCLUDE_NODES")
sbatch -N "$NODES" -p "$JOB_PARTITION" --comment="$JOB_COMMENT" -t "$TIME" \
  "${EXCL[@]}" "${SW[@]}" --export="$EXPORTS" slurm/run_ttq.sbatch
