#!/usr/bin/env bash
# MLCommons submission packaging.
#
# Assembles a submission tarball in the MLCommons training_results layout:
#
#   <submitter>/
#     systems/<system_desc>.json
#     benchmarks/<workload>/implementations/<impl>/
#       README.md
#       Dockerfile
#       config_*.sh
#       run.sub run_and_time.sh
#       <code...>
#     results/<system_desc>/<workload>/
#       result_0.txt
#       result_1.txt
#       ...
#       compliance_checker_log.txt
#
# Runs mlperf_logging.compliance_checker against every result_*.txt as a
# final gate; packaging aborts if any run fails the checker.
#
# This script does NOT submit to MLCommons — it prepares the artefact for a
# PR to mlcommons/training_results_vX.Y.

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
yesno(){ local p="$1" d="${2-y}" v; while :; do read -r -p "$p (y/n) [$d]: " v; v="${v:-$d}"
          case "$v" in [Yy]*) return 0;; [Nn]*) return 1;; esac; done; }

(( BASH_VERSINFO[0] >= 4 )) || die "Bash >= 4 required"
[[ -t 0 ]] || die "TTY required"

command -v jq >/dev/null 2>&1 || warn "jq missing — system_desc will not be validated."

say "Pick workload"
mapfile -t MANIFESTS < <(ls "$WORKLOADS_DIR"/*.manifest.sh 2>/dev/null)
labels=()
for mf in "${MANIFESTS[@]}"; do
    labels+=("$(basename "$mf" .manifest.sh)")
done
sel=$(pick "workload" "${labels[@]}")
# shellcheck disable=SC1090
source "${MANIFESTS[$((sel-1))]}"

SUBMITTER="$(ask_req 'Submitter org name (e.g. DoNnMyTh, MyCo)')"
SYSTEM_DESC="$(ask_req 'System descriptor name (e.g. my_cluster_512xH200)')"
DIVISION="$(ask 'Division (closed|open)' closed)"
CATEGORY="$(ask 'Category (available|preview|rdi)' available)"
REPO_DIR="$(ask_req 'Path to training_results_v5.1 repo')"
[[ -d "$REPO_DIR/$WL_IMPL_SUBDIR" ]] || die "Impl path not found: $REPO_DIR/$WL_IMPL_SUBDIR"
RESULTS_DIR="$(ask_req 'Path to results directory (contains result_*.txt logs)')"
[[ -d "$RESULTS_DIR" ]] || die "Results dir missing: $RESULTS_DIR"
OUT_DIR="$(ask 'Output directory' "$PWD/submission_${WL_NAME}_${SYSTEM_DESC}")"

IMAGE="$(ask 'Image for compliance checker' "mlperf-nvidia:$WL_IMAGE_TAG_BASE")"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "Image not found: $IMAGE"

# -----------------------------------------------------------------
# Validate each result log against mlperf_logging.compliance_checker
# -----------------------------------------------------------------
shopt -s nullglob
mapfile -t RESULT_LOGS < <(ls "$RESULTS_DIR"/result_*.txt 2>/dev/null)
(( ${#RESULT_LOGS[@]} > 0 )) || die "No result_*.txt logs in $RESULTS_DIR"
info "Found ${#RESULT_LOGS[@]} result log(s)."

PASS=0; FAIL=0
mkdir -p "$OUT_DIR"
CHECKER_LOG="$OUT_DIR/compliance_checker_log.txt"
: > "$CHECKER_LOG"
for log in "${RESULT_LOGS[@]}"; do
    info "Checking $(basename "$log")..."
    {
        echo "=== $(basename "$log") ==="
        docker run --rm --ipc=host -v "$(dirname "$log"):/logs:ro" \
            "$IMAGE" bash -lc "
                pip install -q 'mlperf_logging>=3.0.0' 2>/dev/null || true
                python -m mlperf_logging.compliance_checker \
                    --usage training --ruleset 5.1.0 \
                    /logs/$(basename "$log")
            " 2>&1
        echo
    } >> "$CHECKER_LOG"
    if [[ "${PIPESTATUS[0]}" == "0" ]]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done
info "Compliance: $PASS passed, $FAIL failed (see $CHECKER_LOG)"
(( FAIL == 0 )) || { yesno "Continue packaging despite failures?" n || die "Aborted."; }

# -----------------------------------------------------------------
# Build MLCommons layout
# -----------------------------------------------------------------
IMPL_NAME="$(basename "$WL_IMPL_SUBDIR")"
SUB_ROOT="$OUT_DIR/$SUBMITTER"
SYS_DIR="$SUB_ROOT/systems"
BENCH_DIR="$SUB_ROOT/benchmarks/$WL_NAME/implementations/$IMPL_NAME"
RES_DIR="$SUB_ROOT/results/$SYSTEM_DESC/$WL_NAME"

mkdir -p "$SYS_DIR" "$BENCH_DIR" "$RES_DIR"

say "Copying implementation source"
cp -r "$REPO_DIR/$WL_IMPL_SUBDIR/." "$BENCH_DIR/"
say "Copying result logs"
cp "${RESULT_LOGS[@]}" "$RES_DIR/"
cp "$CHECKER_LOG" "$RES_DIR/compliance_checker_log.txt"

SYS_JSON="$SYS_DIR/${SYSTEM_DESC}.json"
if [[ ! -f "$SYS_JSON" ]]; then
    say "Generating system descriptor skeleton at $SYS_JSON"
    cat > "$SYS_JSON" <<EOF
{
  "division": "$DIVISION",
  "submitter": "$SUBMITTER",
  "status": "$CATEGORY",
  "system_name": "$SYSTEM_DESC",
  "number_of_nodes": "FIXME",
  "host_processors_per_node": "FIXME",
  "host_processor_model_name": "FIXME",
  "host_processor_core_count": "FIXME",
  "accelerators_per_node": "FIXME",
  "accelerator_model_name": "FIXME",
  "accelerator_memory_capacity": "FIXME",
  "accelerator_interconnect": "FIXME",
  "host_networking": "FIXME",
  "framework": "FIXME",
  "operating_system": "FIXME"
}
EOF
    info "Edit $SYS_JSON and replace every FIXME before submission."
else
    info "Using existing $SYS_JSON"
fi
[[ -n "$(command -v jq)" ]] && jq . "$SYS_JSON" >/dev/null || warn "JSON validation skipped."

TAR="$OUT_DIR/submission_${WL_NAME}_${SYSTEM_DESC}.tar.gz"
say "Creating tarball $TAR"
( cd "$OUT_DIR" && tar -czf "$(basename "$TAR")" "$SUBMITTER" )
say "Done."
info "Submission tree: $SUB_ROOT"
info "Tarball        : $TAR"
info "Next: open a PR against mlcommons/training_results_v5.1 with this tree."
