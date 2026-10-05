#!/usr/bin/env bash
# Run a command while sampling total GPU memory use with nvidia-smi (every 0.5 s).
# Prints the idle baseline taken just before the command and the peak during it.
# The baseline includes whatever the Windows desktop is using at the time.
set -u
LOG=$(mktemp)
BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$LOG" &
SAMPLER=$!
"$@"
STATUS=$?
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
PEAK=$(cut -d, -f1 "$LOG" | sort -n | tail -1)
UTIL=$(cut -d, -f2 "$LOG" | awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n}')
echo "GPU_MEM baseline_mib=$BASE peak_mib=$PEAK delta_mib=$((PEAK - BASE)) mean_util_pct=$UTIL samples=$(wc -l < "$LOG")"
rm -f "$LOG"
exit "$STATUS"
