#!/usr/bin/env bash
# Post-training evaluation / log inspection tool.
#
# Runs the MLCommons `mlperf_logging.compliance_checker` against a training
# log, extracts the final quality metric vs. the workload target, and prints
# a human-readable summary plus a machine-readable one-line status.

set -u
set -o pipefail

# --- mlperf.sh common-lib hook -----------------------------------------
_MLPERF_LIB_SOURCED=0
if _LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/common.sh" && [[ -f "$_LIB" ]]; then
    # shellcheck source=../lib/common.sh
    source "$_LIB"
    _MLPERF_LIB_SOURCED=1
fi
# Auto-yes / config-file via env only — no flag parsing here to avoid
# clobbering per-tool argv handling.
: "${MLPERF_AUTO_YES:=0}"
if [[ -n "${MLPERF_CONFIG_FILE:-}" && -f "${MLPERF_CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${MLPERF_CONFIG_FILE}"
    MLPERF_AUTO_YES=1
fi
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKLOADS_DIR="$REPO_ROOT/workloads"

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
(( ${#MANIFESTS[@]} > 0 )) || die "No workload manifests found in $WORKLOADS_DIR"
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
        pip install -q 'mlperf_logging==3.1.0' 2>/dev/null || true
        python -m mlperf_logging.compliance_checker \
            --usage training --ruleset 5.1.0 \
            /logs/$LOGFILE || exit 2
    "
rc=$?

say "Parsing final-status events"
# Extract :::MLLOG lines for run_start/run_stop and eval_accuracy/eval_error
target_rule="${TARGET:-}"
grep -E ':::MLLOG' "$LOG" | awk -F ':::MLLOG' '{print $2}' | \
    TARGET="$target_rule" python - <<'PY'
import json, os, re, sys
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

# Compare to workload quality target, e.g. "log_perplexity<=3.3" or "mAP>=0.34"
rule = os.environ.get("TARGET", "").strip()
m = re.match(r'^[A-Za-z_]+(<=|>=)([0-9.eE+\-]+)$', rule)
if m and lastacc is not None:
    op, thresh = m.group(1), float(m.group(2))
    try:
        val = float(lastacc)
        met = (val <= thresh) if op == "<=" else (val >= thresh)
        print(f"quality target   : {rule}  (last={val})  → {'MET' if met else 'NOT MET'}")
        sys.exit(0 if met else 3)
    except (TypeError, ValueError):
        print(f"quality target   : {rule}  (last value not numeric)")
PY
quality_rc=$?

say "RESULT"
if (( rc == 0 )); then
    info "compliance check: PASS"
else
    info "compliance check: FAIL (exit $rc)"
fi
case "$quality_rc" in
    0) info "quality target : MET" ;;
    3) info "quality target : NOT MET" ;;
    *) info "quality target : not evaluated (no numeric match)" ;;
esac

final=0
(( rc == 0 )) || final=1
# quality_rc: 0 = met, 3 = not met (fail), anything else (e.g. no rule parsed,
# or non-numeric lastacc) is treated as "not evaluated" and does not flip
# the overall result.
(( quality_rc == 3 )) && final=1
if (( final == 0 )); then echo "STATUS=${WL_NAME}:PASS"; else echo "STATUS=${WL_NAME}:FAIL"; fi
exit "$final"
