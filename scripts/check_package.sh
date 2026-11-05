#!/bin/bash
# Run the MLPerf package checker (training, ruleset 6.1.0 by default) on submission/<SUBMITTER>.
# The checker writes scaling.json (RCP normalization) into each result directory; keep it in the
# final package. Old scaling.json files are removed first so reruns do not warn.
cd "$(dirname "$0")/.."
source ./env.sh
RULES=${1:-6.1.0}
cd "$SUBMISSION_DIR" || exit 1
find "$SUBMITTER/results" -name scaling.json -delete
rm -f package_checker.log
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYCHECK" -m mlperf_logging.package_checker "$SUBMITTER" training "$RULES" > /dev/null 2>&1
grep -E "System Hangang|normalization factor|RCP Test (Passed|Failed)|did not converge|FOUND|ERROR" package_checker.log
