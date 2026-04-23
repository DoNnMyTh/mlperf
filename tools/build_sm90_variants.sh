#!/usr/bin/env bash
# Build + push -sm90 image variants for every NeMo workload that declares
# an NVTE_CUDA_ARCHS-style WL_DOCKERFILE_PATCH_TO in its manifest.
#
# Why it exists: upstream Dockerfiles default to Blackwell-only sm_100/103.
# This wrapper rewrites the NVTE_CUDA_ARCHS line via idempotent regex
# replace, builds, tags, and pushes. nvcc cross-compiles sm_90 from any
# x86_64 host — no Hopper GPU required to produce Hopper-runnable images.
#
# Usage:
#   tools/build_sm90_variants.sh \
#       --repo /path/to/training_results_v5.1 \
#       --hub  donnmyth/mlperf-nvidia \
#       [--logdir /tmp/mlperf-build-logs] \
#       [--workloads "llama31_8b llama31_405b ..."]
#
# Env overrides: REPO, HUB, LOGDIR, MLPERF_REPO.

set -u
export MSYS_NO_PATHCONV=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLPERF_REPO="${MLPERF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# shellcheck source=../lib/common.sh
if [[ -f "$SCRIPT_DIR/../lib/common.sh" ]]; then source "$SCRIPT_DIR/../lib/common.sh"; fi

REPO="${REPO:-}"
HUB="${HUB:-donnmyth/mlperf-nvidia}"
LOGDIR="${LOGDIR:-/tmp/mlperf-build-logs}"
WORKLOADS_FILTER=""

while (( $# )); do
    case "$1" in
        --repo)      shift; REPO="$1" ;;
        --hub)       shift; HUB="$1" ;;
        --logdir)    shift; LOGDIR="$1" ;;
        --workloads) shift; WORKLOADS_FILTER="$1" ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -d "$REPO" ]] || { echo "--repo required (clone of mlcommons/training_results_vX.Y)" >&2; exit 1; }

mkdir -p "$LOGDIR"
STATUS="$LOGDIR/build_sm90.status"
: > "$STATUS"

# Discover every manifest declaring an NVTE_CUDA_ARCHS-style patch.
declare -a SELECTED=()
for m in "$MLPERF_REPO"/workloads/*.manifest.sh; do
    name=$(basename "$m" .manifest.sh)
    if [[ -n "$WORKLOADS_FILTER" ]] && ! [[ " $WORKLOADS_FILTER " == *" $name "* ]]; then
        continue
    fi
    patch_to=$(bash -c "set -u; source '$m' && echo \"\${WL_DOCKERFILE_PATCH_TO:-}\"")
    [[ "$patch_to" == NVTE_CUDA_ARCHS=\"*\" ]] || continue
    SELECTED+=("$name")
done
(( ${#SELECTED[@]} > 0 )) || { say "no eligible workloads (need NVTE_CUDA_ARCHS patch in manifest)"; exit 1; }
say "eligible workloads: ${SELECTED[*]}"

# Regex-replace the NVTE_CUDA_ARCHS line in Dockerfile. Idempotent —
# works from any prior state (upstream default, sm89, sm90, etc.).
set_nvte_archs() {
    local dir="$1" target_line="$2"
    ( cd "$dir" || exit 1
      if grep -qE 'NVTE_CUDA_ARCHS="[^"]*"' Dockerfile; then
          sed -i -E "s|NVTE_CUDA_ARCHS=\"[^\"]*\"|$target_line|" Dockerfile
          grep -oE 'NVTE_CUDA_ARCHS="[^"]*"' Dockerfile | head -1
      else
          echo "no NVTE_CUDA_ARCHS line"
      fi
    )
}

for wl in "${SELECTED[@]}"; do
    manifest="$MLPERF_REPO/workloads/$wl.manifest.sh"
    impl_subdir=$(bash -c "set -u; source '$manifest' && echo \"\$WL_IMPL_SUBDIR\"")
    base_tag=$(bash -c    "set -u; source '$manifest' && echo \"\$WL_IMAGE_TAG_BASE\"")
    target_line=$(bash -c "set -u; source '$manifest' && echo \"\$WL_DOCKERFILE_PATCH_TO\"")

    dir="$REPO/$impl_subdir"
    sm90_tag="$base_tag-sm90"
    local_img="mlperf-nvidia:$sm90_tag"
    remote_img="$HUB:$sm90_tag"
    log="$LOGDIR/${wl}_sm90.log"

    say "=== $wl -> $sm90_tag ==="
    [[ -d "$dir" ]] || { say "  skip — $dir not found"; continue; }

    say "  set Dockerfile: $target_line"
    r=$(set_nvte_archs "$dir" "$target_line"); say "  now: $r"

    say "  docker build $local_img"
    ( cd "$dir" && docker build -t "$local_img" . ) > "$log" 2>&1
    rc=$?
    if (( rc != 0 )); then
        say "  BUILD FAIL ($rc) — see $log"
        tail -15 "$log" >> "$STATUS"
        continue
    fi

    docker tag "$local_img" "$remote_img"
    say "  push $remote_img"
    docker push "$remote_img" >> "$log" 2>&1 || { say "  PUSH FAIL"; continue; }
    say "  OK $wl -> $remote_img"
done

say "ALL DONE"
