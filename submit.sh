#!/bin/bash
# Submit a TTQ run.
#   ./submit.sh NODES [KEY=VALUE ...]
# Keys:
#   MINIBS CP LR MAX_STEPS CP_COMM_TYPE   -> override KRAI config values
#   NETMODE=C|A   (default C = Slingshot CXI, A = TCP sockets)
#   SEED=<int>    NCCL_DEBUG=INFO|WARN    TIME=HH:MM:SS
# Examples (best configurations found on Hangang):
#   ./submit.sh 1
#   ./submit.sh 8
#   ./submit.sh 16                      # KRAI n16 config: TP4 CP2 MINIBS1 -> GBS 8
#   ./submit.sh 32 MINIBS=1 CP=2        # GBS 16
#   ./submit.sh 64 MINIBS=1 CP=2        # GBS 32
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
NODES=${1:?usage: ./submit.sh NODES [KEY=VALUE ...]}; shift

case "$PWD" in /scratch/*|/appsdata/*) ;; *) echo "Submit from /scratch or /appsdata (Hangang policy)"; exit 1;; esac

EXPORTS="ALL"; TIME=01:30:00
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  case "$k" in
    MINIBS|CP|LR|MAX_STEPS|CP_COMM_TYPE) EXPORTS="$EXPORTS,OV_$k=$v" ;;
    NETMODE|SEED|NCCL_DEBUG)             EXPORTS="$EXPORTS,$k=$v" ;;
    TIME)                                TIME=$v ;;
    *) echo "Unknown key: $k"; exit 1 ;;
  esac
done

mkdir -p "$RESULTS"
EXCL=(); [ -n "$EXCLUDE_NODES" ] && EXCL=(--exclude="$EXCLUDE_NODES")
set -x
sbatch -N "$NODES" -p "$SLURM_PARTITION" --comment="$SLURM_COMMENT" -t "$TIME" \
  "${EXCL[@]}" --export="$EXPORTS" slurm/run_ttq.sbatch
