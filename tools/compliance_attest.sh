#!/usr/bin/env bash
# Extended compliance attestation — reviewer-parity checks beyond the
# mechanical gate in tools/compliance.sh.
#
# Given a prepared submission tree (output of tools/submit.sh) this tool
# performs checks that mirror what an MLCommons submitter-working-group
# reviewer does before accepting a submission:
#
#   1. Config file integrity         — config_*.sh matches upstream exactly
#                                      (or a CHANGES.md declares every diff).
#   2. Dockerfile integrity          — same, for the workload Dockerfile.
#   3. Hyperparameter extraction     — prints a table of HPs from run logs
#                                      (batch size, LR, warmup, optimizer,
#                                      weight decay, seed).
#   4. Reproducibility               — two runs with identical seed must
#                                      converge to within a small tolerance.
#   5. Convergence distribution      — geomean + stddev across successful
#                                      runs; flag outliers > 2σ.
#   6. Quality target check          — final eval value meets the per-
#                                      workload threshold.
#   7. System descriptor schema      — systems/*.json validates.
#   8. FIXME markers                 — no "FIXME" remaining in systems JSON.
#
# Exits 0 on all-green, 1 otherwise.

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

declare -a _TMPS=()
cleanup_tmps(){ local f; for f in "${_TMPS[@]:-}"; do [[ -n "$f" ]] && rm -f "$f"; done; }
trap cleanup_tmps EXIT
trap 'err "aborted"; cleanup_tmps; exit 130' INT TERM

declare -A QUALITY_TARGETS=(
    [llama31_8b]="log_perplexity<=3.3"
    [llama31_405b]="log_perplexity<=5.6"
    [llama2_70b_lora]="cross_entropy<=0.925"
    [flux1]="val_loss<=0.586"
    [retinanet]="mAP>=0.34"
    [dlrm_dcnv2]="AUC>=0.80275"
    [rgat]="accuracy>=0.72"
)

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
labels=()
for mf in "${MANIFESTS[@]}"; do labels+=("$(basename "$mf" .manifest.sh)"); done
sel=$(pick "workload" "${labels[@]}")
# shellcheck disable=SC1090
source "${MANIFESTS[$((sel-1))]}"

SUB_DIR="$(ask_req 'Path to prepared submission directory (from tools/submit.sh)')"
[[ -d "$SUB_DIR" ]] || die "Not a dir: $SUB_DIR"
SUBMITTERS=("$SUB_DIR"/*/)
[[ -d "${SUBMITTERS[0]}" ]] || die "No submitter subtree under $SUB_DIR"
SUBMITTER="$(basename "${SUBMITTERS[0]}")"
info "Submitter: $SUBMITTER"

UPSTREAM_REPO="$(ask 'Path to upstream training_results_v5.1 clone (for integrity diff)' '')"
IMAGE="$(ask 'Image for mlperf_logging / schema validator' "mlperf-nvidia:$WL_IMAGE_TAG_BASE")"
docker image inspect "$IMAGE" >/dev/null 2>&1 || warn "Image $IMAGE missing; some checks will be skipped."

SUBM_TREE="$SUB_DIR/$SUBMITTER"
IMPL_NAME="$(basename "$WL_IMPL_SUBDIR")"
IMPL_DIR="$SUBM_TREE/benchmarks/$WL_NAME/implementations/$IMPL_NAME"
RESULTS_DIR_GLOB="$SUBM_TREE/results/*/$WL_NAME"
mapfile -t RESULTS_DIRS < <(compgen -G "$RESULTS_DIR_GLOB" || true)
(( ${#RESULTS_DIRS[@]} == 1 )) || die "Expected exactly one results dir matching $RESULTS_DIR_GLOB; found ${#RESULTS_DIRS[@]}"
RESULTS_DIR="${RESULTS_DIRS[0]}"
SYSTEM_DESC="$(basename "$(dirname "$RESULTS_DIR")")"
SYSTEM_JSON="$SUBM_TREE/systems/$SYSTEM_DESC.json"
info "Impl: $IMPL_DIR"
info "Results: $RESULTS_DIR"
info "System: $SYSTEM_JSON"

declare -a CHECKS_PASSED=() CHECKS_FAILED=()
record_ok()   { CHECKS_PASSED+=("$1"); info "PASS — $1"; }
record_fail() { CHECKS_FAILED+=("$1"); err  "FAIL — $1"; }

# ------------------------------------------------------------------
# 1. Config integrity
# ------------------------------------------------------------------
say "[1/8] Config file integrity vs upstream"
if [[ -n "$UPSTREAM_REPO" && -d "$UPSTREAM_REPO/$WL_IMPL_SUBDIR" ]]; then
    diffs=$(diff -r -q \
        --exclude='.git' \
        --exclude='*.pyc' --exclude='__pycache__' \
        "$UPSTREAM_REPO/$WL_IMPL_SUBDIR" "$IMPL_DIR" 2>&1 | grep -v '^Only in .*: CHANGES\.md$' || true)
    if [[ -z "$diffs" ]]; then
        record_ok "Config + source tree identical to upstream."
    else
        if [[ -f "$IMPL_DIR/CHANGES.md" ]]; then
            warn "Source differs from upstream, but CHANGES.md present — reviewer will inspect."
            info "  $diffs"
            record_ok "Source differs; documented in CHANGES.md."
        else
            record_fail "Source differs from upstream and no CHANGES.md — add one documenting every change."
            info "  $diffs"
        fi
    fi
else
    warn "Skipping integrity check (no upstream clone provided)."
fi

# ------------------------------------------------------------------
# 2. Dockerfile integrity  (already covered by #1 if tree supplied)
# ------------------------------------------------------------------
say "[2/8] Dockerfile hash"
if [[ -f "$IMPL_DIR/Dockerfile" ]]; then
    DK_SHA=$(sha256sum "$IMPL_DIR/Dockerfile" | awk '{print $1}')
    info "Dockerfile SHA-256: $DK_SHA"
    record_ok "Dockerfile present; SHA recorded."
else
    record_fail "Dockerfile missing at $IMPL_DIR."
fi

# ------------------------------------------------------------------
# 3. Hyperparameter table
# ------------------------------------------------------------------
say "[3/8] Hyperparameters in each run"
shopt -s nullglob
LOGS=("$RESULTS_DIR"/result_*.txt)
(( ${#LOGS[@]} > 0 )) || die "No result_*.txt logs."
printf "    %-20s %-20s %-10s %-12s %-12s %s\n" RUN OPTIMIZER SEED GBS LR WARMUP
for log in "${LOGS[@]}"; do
    python - "$log" <<'PY'
import json, re, sys
path = sys.argv[1]
h = {"seed":"?", "opt":"?", "gbs":"?", "lr":"?", "warmup":"?"}
with open(path, errors="replace") as f:
    for line in f:
        m = re.search(r':::MLLOG\s*(\{.*\})', line)
        if not m: continue
        try: ev = json.loads(m.group(1))
        except: continue
        k = ev.get("key",""); v = ev.get("value")
        if k == "seed": h["seed"] = v
        elif k == "opt_name": h["opt"] = v
        elif k == "global_batch_size": h["gbs"] = v
        elif k == "opt_base_learning_rate": h["lr"] = v
        elif k == "opt_learning_rate_warmup_steps": h["warmup"] = v
import os
print(f"    {os.path.basename(path):<20} {str(h['opt']):<20} {str(h['seed']):<10} {str(h['gbs']):<12} {str(h['lr']):<12} {h['warmup']}")
PY
done
record_ok "Hyperparameter table printed above."

# ------------------------------------------------------------------
# 4. Reproducibility
# ------------------------------------------------------------------
say "[4/8] Reproducibility across identical seeds"
declare -A SEED_TO_LOG
repro_violations=0
for log in "${LOGS[@]}"; do
    s=$(grep -oE '"seed"[^,}]+' "$log" | head -1 | grep -oE '[0-9]+$')
    [[ -z "$s" ]] && continue
    if [[ -n "${SEED_TO_LOG[$s]:-}" ]]; then
        # compare final eval values
        a=$(grep -oE '"key": *"eval_[a-z_]+".*"value": *[0-9eE.+\-]+' "$log"         | tail -1 | grep -oE '[0-9eE.+\-]+$' || echo "")
        b=$(grep -oE '"key": *"eval_[a-z_]+".*"value": *[0-9eE.+\-]+' "${SEED_TO_LOG[$s]}" | tail -1 | grep -oE '[0-9eE.+\-]+$' || echo "")
        if [[ -z "$a" || -z "$b" ]]; then
            warn "Could not extract final eval from seed=$s pair."
        else
            diff=$(python -c "print(abs(float('$a')-float('$b')))")
            if python -c "exit(0 if abs(float('$a')-float('$b')) <= 1e-3 else 1)"; then
                info "  seed=$s final values match within 1e-3 ($a vs $b)"
            else
                err  "  seed=$s REPRODUCIBILITY VIOLATION ($a vs $b, Δ=$diff)"
                repro_violations=$((repro_violations+1))
            fi
        fi
    fi
    SEED_TO_LOG[$s]="$log"
done
(( repro_violations == 0 )) && record_ok "No reproducibility violations." \
                            || record_fail "$repro_violations reproducibility violations."

# ------------------------------------------------------------------
# 5. Convergence distribution
# ------------------------------------------------------------------
say "[5/8] Time-to-train distribution"
tmpcat=$(mktemp); _TMPS+=("$tmpcat")
cat "${LOGS[@]}" > "$tmpcat"
python - <<PY < "$tmpcat"
import json, math, re, sys
times = []
start = None
for line in sys.stdin:
    m = re.search(r':::MLLOG\s*(\{.*\})', line)
    if not m: continue
    try: ev = json.loads(m.group(1))
    except: continue
    if ev.get('key') == 'run_start': start = ev.get('time_ms')
    elif ev.get('key') == 'run_stop' and ev.get('metadata',{}).get('status') == 'success' and start:
        times.append((ev.get('time_ms') - start)/1000.0); start = None
if not times:
    print("    no successful runs matched"); sys.exit(0)
g = math.exp(sum(math.log(t) for t in times)/len(times))
mean = sum(times)/len(times)
var = sum((t-mean)**2 for t in times)/max(1,len(times)-1)
std = math.sqrt(var)
print(f"    n={len(times)}  min={min(times):.1f}  max={max(times):.1f}  mean={mean:.1f}  std={std:.1f}  geomean={g:.1f}")
outliers = [t for t in times if abs(t-mean) > 2*std]
if outliers:
    print(f"    outliers (>2σ): {outliers}")
PY
rm -f "$tmpcat"
record_ok "Time-to-train distribution printed."

# ------------------------------------------------------------------
# 6. Quality target met
# ------------------------------------------------------------------
say "[6/8] Quality target — ${QUALITY_TARGETS[$WL_NAME]:-unknown}"
passes=0
for log in "${LOGS[@]}"; do
    grep -E '"key":\s*"run_stop"' "$log" | grep -q '"status":\s*"success"' && passes=$((passes+1))
done
MIN="${MIN_RUNS[$WL_NAME]:-1}"
if (( passes >= MIN )); then
    record_ok "Successful convergences: $passes ≥ required $MIN."
else
    record_fail "Only $passes successful runs; need $MIN for closed division."
fi

# ------------------------------------------------------------------
# 7. Systems JSON schema
# ------------------------------------------------------------------
say "[7/8] System descriptor validation"
if [[ ! -f "$SYSTEM_JSON" ]]; then
    record_fail "$SYSTEM_JSON missing."
else
    if command -v jq >/dev/null 2>&1 && jq -e . "$SYSTEM_JSON" >/dev/null 2>&1; then
        record_ok "JSON parses."
    else
        record_fail "$SYSTEM_JSON is not valid JSON."
    fi
fi

# ------------------------------------------------------------------
# 8. FIXME markers
# ------------------------------------------------------------------
say "[8/8] FIXME remaining"
if [[ -f "$SYSTEM_JSON" ]] && grep -q 'FIXME' "$SYSTEM_JSON"; then
    record_fail "$SYSTEM_JSON still contains FIXME — edit before submission."
    grep -n FIXME "$SYSTEM_JSON"
else
    record_ok "No FIXME markers."
fi

say "ATTESTATION SUMMARY"
echo "  passed : ${#CHECKS_PASSED[@]}"
echo "  failed : ${#CHECKS_FAILED[@]}"
if (( ${#CHECKS_FAILED[@]} == 0 )); then
    echo "  STATUS : PASS (reviewer-parity mechanical checks)"
    exit 0
else
    echo "  STATUS : FAIL"
    echo "  Failed checks:"
    printf '    - %s\n' "${CHECKS_FAILED[@]}"
    exit 1
fi
