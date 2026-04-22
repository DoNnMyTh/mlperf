#!/usr/bin/env bash
# Scan a local clone of mlcommons/training_results_vX.Y for unknown NVIDIA
# workloads and emit manifest stubs into workloads/.
#
# A "new" workload is detected as:
#   NVIDIA/benchmarks/<name>/implementations/<any-impl>/Dockerfile
# where workloads/<name>.manifest.sh does not yet exist in this repo.
#
# The generated stub has placeholders flagged with FIXME; the operator must
# fill in dataset layout, mounts, smoke env, etc.

set -u
set -o pipefail

_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/common.sh"
[[ -f "$_LIB" ]] && source "$_LIB"

REPO=""; OUT_DIR=""
while (( $# )); do
    case "$1" in
        --repo) shift; REPO="$1" ;;
        --out)  shift; OUT_DIR="$1" ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -d "$REPO" ]] || { echo "--repo required" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/workloads}"

mkdir -p "$OUT_DIR"
new=0
for d in "$REPO/NVIDIA/benchmarks/"*/; do
    name=$(basename "$d")
    [[ -e "$OUT_DIR/${name}.manifest.sh" ]] && continue
    impl=$(ls -d "$d/implementations/"*/ 2>/dev/null | head -1)
    [[ -d "$impl" ]] || continue
    rel="${impl#$REPO/}"
    rel="${rel%/}"
    cat > "$OUT_DIR/${name}.manifest.sh" <<EOF
# Workload manifest — ${name} (auto-generated stub)
# Review every FIXME before using. Compare to workloads/llama31_8b.manifest.sh.

WL_NAME="${name}"
WL_DISPLAY="${name} (FIXME: human name)"
WL_IMPL_SUBDIR="${rel}"
WL_HUB_REPO=""
WL_IMAGE_TAG_BASE="${name}-pyt"     # FIXME: match upstream image name
WL_IMAGE_TAG_VARIANTS=()

WL_DATASET_SUBDIR="${name}"         # FIXME
WL_DATASET_SIZE_GB=100              # FIXME
WL_DATASET_MARKER_FILES=()          # FIXME
WL_DATASET_MARKER_DIRS=()
WL_DOWNLOAD_SCRIPT=""               # FIXME (e.g. data_scripts/download.sh)
WL_DOWNLOAD_ENV=""                  # FIXME (e.g. "DATADIR=/data/foo")
WL_DOWNLOAD_HOST_ENV=""

WL_PREPROC_HOST_SUBPATH="${name}"   # FIXME
WL_PREPROC_MOUNT="/preproc_data"    # FIXME
WL_TOKENIZER_HOST_SUBPATH=""
WL_TOKENIZER_MOUNT=""

WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY=""                   # FIXME if multi-node torchrun needed
WL_CONTAINER_WORKDIR="/workspace"    # FIXME

WL_SMOKE_SUPPORTED=0
WL_SMOKE_ENV=()
WL_SMOKE_PROMPTS=()

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/${rel}"
WL_DOCKERFILE_PATCH_FROM=""
WL_DOCKERFILE_PATCH_TO=""
EOF
    echo "generated $OUT_DIR/${name}.manifest.sh"
    new=$((new+1))
done

if (( new == 0 )); then
    echo "No new workloads detected under $REPO/NVIDIA/benchmarks/."
else
    echo "Generated $new stub(s). Review every FIXME before committing."
fi
