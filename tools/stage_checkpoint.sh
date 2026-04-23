#!/usr/bin/env bash
# Stage a pretrained checkpoint end-to-end for LLM workloads.
#
# For workloads that resume from a published HF checkpoint (Llama 3.1 405B,
# Llama 2 70B LoRA), the full pipeline is:
#   1. huggingface-cli download <repo> --local-dir <host>/hf_ckpt
#   2. convert HF -> NeMo distributed checkpoint
#   3. leave it under <host>/ckpt/<workload>/ in the layout run.sub expects
#
# This tool wraps those three steps. Conversion is delegated to
# tools/convert_weights.sh which runs inside the workload container.
#
# Usage:
#   MLPERF_AUTO_YES=1 \
#     HF_REPO=meta-llama/Llama-3.1-405B \
#     HF_TOKEN=hf_xxx \
#     CKPT_DIR=/data/ckpt \
#     bash tools/stage_checkpoint.sh --workload llama31_405b

set -u
set -o pipefail

_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/common.sh"
[[ -f "$_LIB" ]] && source "$_LIB"

WL=""
while (( $# )); do
    case "$1" in
        --workload) shift; WL="$1" ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -n "$WL" ]] || { echo "--workload required (e.g. llama31_405b)" >&2; exit 1; }

: "${HF_REPO:?HF_REPO env var required (e.g. meta-llama/Llama-3.1-405B)}"
: "${CKPT_DIR:?CKPT_DIR env var required — host path where hf_ckpt/ and ckpt/ will live}"

mkdir -p "$CKPT_DIR/hf_ckpt" "$CKPT_DIR/ckpt/$WL"

echo "[1/3] Downloading $HF_REPO -> $CKPT_DIR/hf_ckpt"
if command -v huggingface-cli >/dev/null 2>&1; then
    HF_TOKEN="${HF_TOKEN:-}" huggingface-cli download "$HF_REPO" \
        --local-dir "$CKPT_DIR/hf_ckpt" --local-dir-use-symlinks False
else
    # Fallback: python -c 'from huggingface_hub import snapshot_download; ...'
    python3 - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["HF_REPO"],
                  local_dir=os.environ["CKPT_DIR"] + "/hf_ckpt",
                  token=os.environ.get("HF_TOKEN") or None,
                  local_dir_use_symlinks=False)
PY
fi

echo "[2/3] Converting HF -> NeMo for workload '$WL' (direct docker invocation)"
# Previously this delegated to tools/convert_weights.sh with MLPERF_AUTO_YES,
# but AUTO_YES picks the FIRST manifest alphabetically (dlrm_dcnv2) — not
# the workload the operator passed via --workload. Run the conversion
# directly here so --workload actually wires through.
MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../workloads" && pwd)/$WL.manifest.sh"
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
# shellcheck disable=SC1090
source "$MANIFEST"
IMAGE="${IMAGE:-mlperf-nvidia:${WL_IMAGE_TAG_BASE}}"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || { echo "Image $IMAGE not found. Pull or build first." >&2; exit 1; }
case "$WL" in
    llama31_8b|llama31_405b)
        SIZE="${WL_DATASET_SUBDIR}"
        docker run --rm --gpus all --ipc=host --shm-size=8g \
            -v "$CKPT_DIR/hf_ckpt:/src:ro" \
            -v "$CKPT_DIR/ckpt/$WL:/dst" \
            -e SIZE="$SIZE" \
            "$IMAGE" bash -c '
                set -e
                python -c "
from nemo.collections.llm import import_ckpt
from nemo.collections.llm.gpt.model.llama import LlamaModel, Llama31Config${SIZE^^}
cfg = Llama31Config${SIZE^^}()
model = LlamaModel(config=cfg)
import_ckpt(model=model, source=\"hf:///src\", output_path=\"/dst\")
"'
        ;;
    llama2_70b_lora)
        docker run --rm --gpus all --ipc=host --shm-size=16g \
            -v "$CKPT_DIR/hf_ckpt:/src:ro" \
            -v "$CKPT_DIR/ckpt/$WL:/dst" \
            "$IMAGE" bash -c 'cd /workspace/ft-llm && python scripts/convert_model.py --output_path /dst --source /src'
        ;;
    *)
        echo "stage_checkpoint: no conversion defined for $WL" >&2
        exit 1 ;;
esac

echo "[3/3] Layout OK under $CKPT_DIR/ckpt/$WL"
ls -la "$CKPT_DIR/ckpt/$WL" 2>/dev/null || echo "(empty — conversion may require manual review)"
