#!/usr/bin/env bash
# MLPerf Training v5.1 compliance certification helper.
#
# Accepts a directory containing result_*.txt logs from an `sbatch run.sub`
# sweep and verifies the following, in order:
#
#   1. Every file passes mlperf_logging.compliance_checker 5.1.0.
#   2. Every run has a distinct seed.
#   3. The number of successful convergences meets the minimum per workload
#      (MLPerf defines per-benchmark run counts for reporting).
#   4. A geomean time-to-train is computed across the successful runs and
#      compared to a user-supplied reference or printed bare.
#   5. SLURM_JOB_ID metadata is present (heuristic for "the run was launched
#      via sbatch" — required for closed-division submission).
#
# This does NOT prove compliance by itself — MLCommons requires the full
# submission package to be reviewed by the submitter working group. The tool
# enforces the mechanical checks that operators can run locally.

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

# Clean up any temp files we create on exit/abort.
declare -a _TMPS=()
cleanup_tmps(){ local f; for f in "${_TMPS[@]:-}"; do [[ -n "$f" ]] && rm -f "$f"; done; }
trap cleanup_tmps EXIT
trap 'err "aborted"; cleanup_tmps; exit 130' INT TERM

# MLPerf training v5.1 minimum number of convergence runs per benchmark.
# Reference: https://github.com/mlcommons/training_policies
declare -A MIN_RUNS=(
    [llama31_8b]=3
    [llama31_405b]=3
    [llama2_70b_lora]=10
    [flux1]=3
    [retinanet]=5
    [dlrm_dcnv2]=5
    [rgat]=10
)

say "Pick workload"
mapfile -t MANIFESTS < <(ls "$WORKLOADS_DIR"/*.manifest.sh 2>/dev/null)
(( ${#MANIFESTS[@]} > 0 )) || die "No workload manifests found in $WORKLOADS_DIR"
labels=()
for mf in "${MANIFESTS[@]}"; do
    n="$(basename "$mf" .manifest.sh)"
    labels+=("$n (min runs: ${MIN_RUNS[$n]:-?})")
done
sel=$(pick "workload" "${labels[@]}")
# shellcheck disable=SC1090
source "${MANIFESTS[$((sel-1))]}"

RESULTS_DIR="$(ask_req 'Path to results directory (result_*.txt)')"
[[ -d "$RESULTS_DIR" ]] || die "Results dir not found: $RESULTS_DIR"

IMAGE="$(ask 'Image for mlperf_logging' "mlperf-nvidia:$WL_IMAGE_TAG_BASE")"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "Image $IMAGE not found"

shopt -s nullglob
mapfile -t LOGS < <(ls "$RESULTS_DIR"/result_*.txt 2>/dev/null)
(( ${#LOGS[@]} > 0 )) || die "No result_*.txt in $RESULTS_DIR"
info "Found ${#LOGS[@]} log(s)."

# ----------------------------------------------------------------------
# 1. mlperf_logging compliance_checker per run
# ----------------------------------------------------------------------
say "[1/5] compliance_checker"
declare -a PASSED=() FAILED=()
for log in "${LOGS[@]}"; do
    if docker run --rm --ipc=host -v "$(dirname "$log"):/l:ro" "$IMAGE" bash -lc "
        pip install -q 'mlperf_logging>=3.0.0' 2>/dev/null || true
        python -m mlperf_logging.compliance_checker --usage training --ruleset 5.1.0 /l/$(basename "$log")
    " >/dev/null 2>&1; then
        PASSED+=("$log")
    else
        FAILED+=("$log")
    fi
done
info "  passed: ${#PASSED[@]}   failed: ${#FAILED[@]}"
(( ${#FAILED[@]} == 0 )) || { err "Failed runs:"; printf '    %s\n' "${FAILED[@]}" >&2; }

# ----------------------------------------------------------------------
# 2. distinct seeds
# ----------------------------------------------------------------------
say "[2/5] distinct seeds"
declare -A SEEN=()
dupes=0
for log in "${PASSED[@]}"; do
    seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$log" | head -1 | grep -oE '[0-9]+$')
    if [[ -z "$seed" ]]; then
        seed=$(grep -oE 'SEED=[0-9]+' "$log" | head -1 | grep -oE '[0-9]+$')
    fi
    if [[ -z "$seed" ]]; then
        warn "  no seed found in $(basename "$log")"
        continue
    fi
    if [[ -n "${SEEN[$seed]:-}" ]]; then
        err "  seed $seed duplicated (${SEEN[$seed]} and $(basename "$log"))"
        dupes=$((dupes+1))
    fi
    SEEN[$seed]="$(basename "$log")"
done
info "  unique seeds: ${#SEEN[@]}   duplicates: $dupes"

# ----------------------------------------------------------------------
# 3. minimum successful convergences
# ----------------------------------------------------------------------
say "[3/5] convergence count"
SUCC=0
for log in "${PASSED[@]}"; do
    grep -E '"key":[[:space:]]*"run_stop"' "$log" | grep -q '"status":[[:space:]]*"success"' && SUCC=$((SUCC+1))
done
MIN="${MIN_RUNS[$WL_NAME]:-1}"
info "  successful convergences: $SUCC   required: $MIN"
(( SUCC >= MIN )) && info "  OK" || err "  INSUFFICIENT runs for closed-division compliance"

# ----------------------------------------------------------------------
# 4. geomean time-to-train
# ----------------------------------------------------------------------
say "[4/5] time-to-train (seconds, geomean over passing runs)"
python_in=$(mktemp); _TMPS+=("$python_in")
for log in "${PASSED[@]}"; do
    grep -E '"key":[[:space:]]*"run_(start|stop)"' "$log"
    echo '---'
done > "$python_in"
python - <<PY < "$python_in"
import json, math, re, sys
runs = []
start = stop = None
for line in sys.stdin:
    if line.startswith('---'):
        if start and stop:
            runs.append(stop - start)
        start = stop = None
        continue
    m = re.search(r':::MLLOG\s*(\{.*\})', line)
    if not m:
        continue
    try:
        ev = json.loads(m.group(1))
    except Exception:
        continue
    ts = ev.get('time_ms')
    if ev.get('key') == 'run_start':
        start = ts
    elif ev.get('key') == 'run_stop' and ev.get('metadata',{}).get('status')=='success':
        stop = ts
if not runs:
    print("    no matched start/stop pairs")
else:
    times = [r/1000.0 for r in runs]
    g = math.exp(sum(math.log(t) for t in times)/len(times))
    print(f"    n = {len(times)}   min = {min(times):.1f}s   max = {max(times):.1f}s   geomean = {g:.1f}s")
PY
rm -f "$python_in"

# ----------------------------------------------------------------------
# 5. sbatch / Slurm launcher evidence
# ----------------------------------------------------------------------
say "[5/5] Slurm launcher evidence (closed-division heuristic)"
slurm_ok=0
for log in "${PASSED[@]}"; do
    if grep -qE 'SLURM_JOB_ID|slurm\.conf|srun|sbatch' "$log"; then slurm_ok=$((slurm_ok+1)); fi
done
info "  runs showing Slurm evidence: $slurm_ok / ${#PASSED[@]}"
(( slurm_ok == ${#PASSED[@]} )) && info "  OK" || warn "  some runs lack Slurm evidence — closed-division submissions MUST use sbatch run.sub"

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
say "SUMMARY — workload=$WL_NAME"
echo "  compliance_checker pass : ${#PASSED[@]} / ${#LOGS[@]}"
echo "  distinct seeds          : ${#SEEN[@]} (duplicates: $dupes)"
echo "  successful convergences : $SUCC / required $MIN"
echo "  Slurm evidence          : $slurm_ok / ${#PASSED[@]}"

if (( ${#FAILED[@]} == 0 && SUCC >= MIN && dupes == 0 && slurm_ok == ${#PASSED[@]} )); then
    echo "  STATUS                  : PASS (mechanical checks)"
    echo "  Next: run tools/submit.sh to build the submission tarball."
    exit 0
else
    echo "  STATUS                  : FAIL"
    exit 1
fi
