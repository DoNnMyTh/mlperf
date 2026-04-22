#!/usr/bin/env bash
# Build + push all workload docker images.
#
# Serial by default. Set MLPERF_BUILD_PARALLEL=N to run N concurrent builds
# (guard disk pressure — each build needs ~80 GB free).
#
# Requires: a local clone of mlcommons/training_results_v5.1 (passed as
# --repo), a Docker Hub org (--hub), and docker login.

set -u
set -o pipefail

_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/common.sh"
[[ -f "$_LIB" ]] && source "$_LIB"
: "${MLPERF_AUTO_YES:=1}"   # build_all is non-interactive by default.

REPO=""; HUB=""; PARALLEL="${MLPERF_BUILD_PARALLEL:-1}"
while (( $# )); do
    case "$1" in
        --repo)     shift; REPO="$1" ;;
        --hub)      shift; HUB="$1" ;;
        --parallel) shift; PARALLEL="$1" ;;
        --help)
            echo "Usage: $0 --repo <path/to/training_results_v5.1> --hub <dockerhub/user> [--parallel N]"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -d "$REPO" ]] || { echo "--repo required" >&2; exit 1; }
[[ -n "$HUB" ]]  || { echo "--hub required" >&2; exit 1; }

LOGDIR="${LOGDIR:-/tmp/mlperf-build-logs}"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/status.log"
: > "$STATUS"

# Workload entries: name | impl subdir | tag
read -r -d '' ENTRIES <<'EOF' || true
retinanet|NVIDIA/benchmarks/retinanet/implementations/tyche_ngpu8_ngc25.04_pytorch|single_stage_detector-pyt
dlrm_dcnv2|NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr|recommendation-hugectr
rgat|NVIDIA/benchmarks/rgat/implementations/tyche_ngpu8_ngc25.03_dgl|graph_neural_network-dgl
flux1|NVIDIA/benchmarks/flux1/implementations/theia_ngpu16_ngc25.09_nemo|flux1-pyt
llama2_70b_lora|NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo|llama2_70b_lora-pyt
llama31_405b|NVIDIA/benchmarks/llama31_405b/implementations/theia_ngpu512_ngc25.09_nemo|llama31_405b-pyt
llama31_8b|NVIDIA/benchmarks/llama31_8b/implementations/nemo|llama31_8b-pyt
EOF

build_one() {
    local wl="$1" sub="$2" tag="$3"
    local log="$LOGDIR/$wl.log"
    local local_img="mlperf-nvidia:$tag"
    local remote_img="$HUB:$tag"
    local dir="$REPO/$sub"
    echo "[$(date +%H:%M:%S)] === $wl ===  $tag" | tee -a "$STATUS"
    [[ -f "$dir/Dockerfile" ]] || { echo "MISSING Dockerfile: $dir" | tee -a "$STATUS"; return 1; }
    (
        cd "$dir" && \
        docker build \
            --cache-from="$remote_img" \
            -t "$local_img" .
    ) >"$log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        echo "[$(date +%H:%M:%S)] BUILD FAIL ($rc) $wl — see $log" | tee -a "$STATUS"
        return 1
    fi
    docker tag "$local_img" "$remote_img"
    docker push "$remote_img" >>"$log" 2>&1 || {
        echo "[$(date +%H:%M:%S)] PUSH FAIL $wl" | tee -a "$STATUS"
        return 1
    }
    echo "[$(date +%H:%M:%S)] OK $wl -> $remote_img" | tee -a "$STATUS"
}

if (( PARALLEL <= 1 )); then
    while IFS='|' read -r wl sub tag; do
        [[ -z "$wl" ]] && continue
        build_one "$wl" "$sub" "$tag" || true
    done <<<"$ENTRIES"
else
    declare -a PIDS=()
    while IFS='|' read -r wl sub tag; do
        [[ -z "$wl" ]] && continue
        # Cap concurrency
        while (( ${#PIDS[@]} >= PARALLEL )); do
            for i in "${!PIDS[@]}"; do
                kill -0 "${PIDS[$i]}" 2>/dev/null || { wait "${PIDS[$i]}" 2>/dev/null; unset 'PIDS[i]'; }
            done
            PIDS=("${PIDS[@]}")
            (( ${#PIDS[@]} >= PARALLEL )) && sleep 2
        done
        ( build_one "$wl" "$sub" "$tag" ) &
        PIDS+=("$!")
    done <<<"$ENTRIES"
    wait
fi

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$STATUS"
echo "Logs: $LOGDIR/"
