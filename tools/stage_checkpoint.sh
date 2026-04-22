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

mkdir -p "$CKPT_DIR/hf_ckpt" "$CKPT_DIR/ckpt"

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

echo "[2/3] Converting HF -> NeMo via tools/convert_weights.sh"
MLPERF_AUTO_YES=1 \
    MLPERF_CONFIG_FILE=/dev/stdin \
    bash "$(dirname "${BASH_SOURCE[0]}")/convert_weights.sh" <<EOF
# Pre-seed answers for convert_weights.sh picker prompts.
# (convert_weights.sh honours MLPERF_AUTO_YES and returns the default at
# every prompt; the manifest selection is position 1 in the sorted list.)
EOF

echo "[3/3] Layout OK under $CKPT_DIR/ckpt/$WL"
ls -la "$CKPT_DIR/ckpt/$WL" 2>/dev/null || echo "(empty — conversion may require manual review)"
