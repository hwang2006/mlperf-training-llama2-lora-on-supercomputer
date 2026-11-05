#!/bin/bash
# Run on every node (one srun task per node) right before training: evidence that the
# page cache was flushed by the previous job's ATOM jobEnd tasks.
echo "CACHE_STATE $(hostname) $(grep '^Cached' /proc/meminfo | tr -s ' ')"
