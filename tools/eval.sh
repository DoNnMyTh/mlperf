#!/usr/bin/env bash
# Post-training evaluation / log inspection tool.
#
# Runs the MLCommons `mlperf_logging.compliance_checker` against a training
# log, extracts the final quality metric vs. the workload target, and prints
# a human-readable summary plus a machine-readable one-line status.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKLOADS_DIR="$REPO_ROOT/workloads"

say()  { printf "\n==> %s\n" "$*"; }
info() { printf "    %s\n" "$*"; }
warn() { printf "WARN: %s\n" "$*" >&2; }
err()  { printf "ERROR: %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }
ask()  { local p="$1" d="${2-}" v; if [[ -n "$d" ]]; then read -r -p "$p [$d]: " v; echo "${v:-$d}"; else read -r -p "$p: " v; echo "$v"; fi; }
ask_req(){ local p="$1" v; while :; do read -r -p "$p: " v; [[ -n "$v" ]] && { echo "$v"; return; }; err "required"; done; }
pick(){ local p="$1"; shift; local i=1; for o in "$@"; do printf "  [%d] %s\n" "$i" "$o" >&2; i=$((i+1)); done
        local v; while :; do read -r -p "$p [1]: " v; v="${v:-1}"; [[ "$v" =~ ^[0-9]+$ ]] && (( v>=1 && v<=$# )) && { echo "$v"; return; }; done; }

(( BASH_VERSINFO[0] >= 4 )) || die "Bash >= 4 required"
[[ -t 0 ]] || die "TTY required"

# MLPerf training v5.1 rules per workload (benchmark_key : quality_target)
declare -A QUALITY_TARGETS=(
    [llama31_8b]="log_perplexity<=3.3"
    [llama31_405b]="log_perplexity<=5.6"
    [llama2_70b_lora]="cross_entropy<=0.925"
    [flux1]="val_loss<=0.586"
    [retinanet]="mAP>=0.34"
    [dlrm_dcnv2]="AUC>=0.80275"
    [rgat]="accuracy>=0.72"
)

say "Pick workload"
mapfile -t MANIFESTS < <(ls "$WORKLOADS_DIR"/*.manifest.sh 2>/dev/null)
labels=()
for mf in "${MANIFESTS[@]}"; do
    n="$(basename "$mf" .manifest.sh)"
    labels+=("$n — ${QUALITY_TARGETS[$n]:-unknown target}")
done
sel=$(pick "workload" "${labels[@]}")
# shellcheck disable=SC1090
source "${MANIFESTS[$((sel-1))]}"
TARGET="${QUALITY_TARGETS[$WL_NAME]:-}"
info "Quality target: ${TARGET:-UNKNOWN}"

LOG="$(ask_req 'Path to training log (MLLOG-formatted or NeMo stdout)')"
[[ -f "$LOG" ]] || die "Log not found: $LOG"

IMAGE="$(ask 'Image to run checker in (local tag)' "mlperf-nvidia:$WL_IMAGE_TAG_BASE")"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "Image $IMAGE not found locally."

LOGDIR="$(dirname "$LOG")"
LOGFILE="$(basename "$LOG")"

say "Running mlperf_logging compliance checker"
docker run --rm --ipc=host \
    -v "$LOGDIR:/logs:ro" \
    "$IMAGE" bash -lc "
        pip install -q 'mlperf_logging>=3.0.0' 2>/dev/null || true
        python -m mlperf_logging.compliance_checker \
            --usage training --ruleset 5.1.0 \
            /logs/$LOGFILE || exit 2
    "
rc=$?

say "Parsing final-status events"
# Extract :::MLLOG lines for run_start/run_stop and eval_accuracy/eval_error
grep -E ':::MLLOG' "$LOG" | awk -F ':::MLLOG' '{print $2}' | \
    python - <<'PY'
import json, sys, re
events = []
for line in sys.stdin:
    try:
        ev = json.loads(line.strip())
        events.append(ev)
    except Exception:
        continue
starts = [e for e in events if e.get("key") == "run_start"]
stops  = [e for e in events if e.get("key") == "run_stop"]
accs   = [e for e in events if e.get("key") in ("eval_accuracy","eval_error")]
lastacc = accs[-1]["value"] if accs else None
status  = stops[-1].get("metadata", {}).get("status") if stops else None
print(f"run_start events : {len(starts)}")
print(f"run_stop  events : {len(stops)}  (last status = {status})")
print(f"eval events      : {len(accs)}")
print(f"last eval value  : {lastacc}")
PY

if (( rc == 0 )); then
    say "RESULT: compliance check PASSED"
else
    say "RESULT: compliance check FAILED (exit $rc). See checker output above."
fi
echo "STATUS=${WL_NAME}:$([[ $rc == 0 ]] && echo PASS || echo FAIL)"
exit $rc
