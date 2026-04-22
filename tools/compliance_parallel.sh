#!/usr/bin/env bash
# Parallel variant of the compliance_checker invocation in tools/compliance.sh.
#
# Runs mlperf_logging.compliance_checker against N result logs concurrently,
# capping parallelism at MLPERF_PARALLEL (default: half the CPU cores, minimum
# 2, maximum 8). Each worker still spawns its own short-lived container —
# parallelism cuts wall time by ~N/W. Serial path remains in compliance.sh.
#
# Usage:
#   MLPERF_PARALLEL=4 bash tools/compliance_parallel.sh <results-dir> <image>

set -u
set -o pipefail

RESULTS_DIR="${1:?results dir required}"
IMAGE="${2:?image required}"
[[ -d "$RESULTS_DIR" ]] || { echo "not a dir: $RESULTS_DIR" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not found: $IMAGE" >&2; exit 1; }

# Default parallelism: min(8, max(2, cpu/2)).
_ncpu=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MLPERF_PARALLEL="${MLPERF_PARALLEL:-$(( _ncpu > 16 ? 8 : (_ncpu < 4 ? 2 : _ncpu / 2) ))}"

shopt -s nullglob
LOGS=("$RESULTS_DIR"/result_*.txt)
(( ${#LOGS[@]} > 0 )) || { echo "no result_*.txt in $RESULTS_DIR" >&2; exit 1; }
echo "Checking ${#LOGS[@]} logs with concurrency=$MLPERF_PARALLEL"

# Bounded-parallelism worker loop: launch up to N in background, wait for any
# to finish before starting next. Collect results.
declare -a PIDS=()
declare -A PID_LOG=()
declare -A LOG_RC=()

run_one() {
    local log="$1"
    docker run --rm --ipc=host -v "$(dirname "$log"):/l:ro" "$IMAGE" bash -lc "
        pip install -q 'mlperf_logging==3.1.0' 2>/dev/null || true
        python -m mlperf_logging.compliance_checker --usage training --ruleset 5.1.0 /l/$(basename "$log")
    " >/dev/null 2>&1
}

for log in "${LOGS[@]}"; do
    # Block until a slot is free.
    while (( ${#PIDS[@]} >= MLPERF_PARALLEL )); do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}"; rc=$?
                LOG_RC["${PID_LOG[${PIDS[$i]}]}"]="$rc"
                unset 'PIDS[i]' 'PID_LOG[${PIDS[$i]}]'
            fi
        done
        PIDS=("${PIDS[@]}")
        (( ${#PIDS[@]} >= MLPERF_PARALLEL )) && sleep 0.5
    done
    run_one "$log" &
    pid=$!
    PIDS+=("$pid")
    PID_LOG["$pid"]="$log"
done

# Drain.
for pid in "${PIDS[@]}"; do
    wait "$pid"; rc=$?
    LOG_RC["${PID_LOG[$pid]}"]="$rc"
done

pass=0; fail=0
for log in "${LOGS[@]}"; do
    rc="${LOG_RC[$log]:-2}"
    if (( rc == 0 )); then pass=$((pass+1)); else fail=$((fail+1)); fi
    printf "  [%s]  %s\n" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$(basename "$log")"
done
echo "Summary: $pass pass, $fail fail, out of ${#LOGS[@]}"
(( fail == 0 )) && exit 0 || exit 1
