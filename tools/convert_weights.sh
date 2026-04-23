#!/usr/bin/env bash
# Model-weight conversion across NeMo / HuggingFace / distcp formats, per workload.
#
# Runs the conversion inside the workload container so that NeMo, HF, and
# custom converters are available. Supports:
#
#   llama31_8b        : NeMo -> HF              (export trained checkpoint for inference)
#                       HF   -> NeMo            (import HF model to NeMo for resume)
#   llama31_405b      : HF   -> NeMo (distcp)   (Meta 405B HF -> NeMo distributed)
#                       NeMo -> HF              (export)
#   llama2_70b_lora   : HF   -> NeMo            (uses scripts/convert_model.py)
#                       LoRA merge              (merge adapter weights into base)
#
# Non-LLM workloads (retinanet, dlrm_dcnv2, rgat, flux1) have bespoke weight
# handling; this tool explicitly no-ops with an informative message.

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
[[ -t 0 ]] || die "Interactive TTY required"

command -v docker >/dev/null 2>&1 || die "docker required"
[[ -d "$WORKLOADS_DIR" ]] || die "workloads/ missing at $WORKLOADS_DIR"

say "Pick workload"
mapfile -t MANIFESTS < <(ls "$WORKLOADS_DIR"/*.manifest.sh 2>/dev/null)
(( ${#MANIFESTS[@]} > 0 )) || die "No workload manifests found in $WORKLOADS_DIR"
labels=()
for mf in "${MANIFESTS[@]}"; do
    n="$(basename "$mf" .manifest.sh)"
    d="$(grep -E '^WL_DISPLAY=' "$mf" | sed -E 's/^WL_DISPLAY="?([^"]*)"?/\1/')"
    labels+=("$n — $d")
done
sel=$(pick "workload" "${labels[@]}")
# shellcheck disable=SC1090
source "${MANIFESTS[$((sel-1))]}"

IMAGE="$(ask 'Image to run conversion in (local tag)' "mlperf-nvidia:$WL_IMAGE_TAG_BASE")"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "Image $IMAGE not found locally. Build or pull it first."

case "$WL_NAME" in
    llama31_8b|llama31_405b)
        say "NeMo <-> HuggingFace conversion for $WL_DISPLAY"
        DIRS=("HuggingFace -> NeMo (import)" "NeMo -> HuggingFace (export)")
        d=$(pick "direction" "${DIRS[@]}")
        SRC="$(ask_req 'Source checkpoint path (host)')"
        DST="$(ask_req 'Destination path (host)')"
        mkdir -p "$DST"
        if (( d == 1 )); then
            # HF -> NeMo distributed
            SIZE="${WL_DATASET_SUBDIR}"   # "8b" or "405b"
            docker run --rm --gpus all --ipc=host --shm-size=8g \
                -v "$SRC:/src:ro" -v "$DST:/dst" \
                -e SRC=/src -e DST=/dst -e SIZE="$SIZE" \
                "$IMAGE" bash -c '
                    set -e
                    python -c "
from nemo.collections.llm import import_ckpt
from nemo.collections.llm.gpt.model.llama import LlamaModel, Llama31Config${SIZE^^}
import os
cfg = Llama31Config${SIZE^^}()
model = LlamaModel(config=cfg)
import_ckpt(model=model, source=\"hf:///src\", output_path=\"/dst\")
"
                '
        else
            # NeMo -> HF
            docker run --rm --gpus all --ipc=host --shm-size=8g \
                -v "$SRC:/src:ro" -v "$DST:/dst" \
                -e SRC=/src -e DST=/dst \
                "$IMAGE" bash -c '
                    set -e
                    python -c "
from nemo.collections.llm import export_ckpt
export_ckpt(path=\"/src\", target=\"hf\", output_path=\"/dst\")
"
                '
        fi
        ;;

    llama2_70b_lora)
        say "Llama 2 70B — HF -> NeMo + optional LoRA merge"
        DIRS=("HF -> NeMo (scripts/convert_model.py)" "Merge LoRA adapter into base")
        d=$(pick "direction" "${DIRS[@]}")
        if (( d == 1 )); then
            SRC="$(ask_req 'HF 70B ckpt path (host, meta-llama/Llama-2-70B-hf)')"
            DST="$(ask_req 'Output NeMo path (host)')"
            mkdir -p "$DST"
            docker run --rm --gpus all --ipc=host --shm-size=16g \
                -v "$SRC:/src:ro" -v "$DST:/dst" \
                "$IMAGE" bash -c 'cd /workspace/ft-llm && python scripts/convert_model.py --output_path /dst --source /src'
        else
            BASE="$(ask_req 'Base NeMo ckpt path (host)')"
            ADAPTER="$(ask_req 'LoRA adapter path (host)')"
            DST="$(ask_req 'Merged output path (host)')"
            mkdir -p "$DST"
            docker run --rm --gpus all --ipc=host --shm-size=16g \
                -v "$BASE:/base:ro" -v "$ADAPTER:/adapter:ro" -v "$DST:/dst" \
                "$IMAGE" bash -c '
                    set -e
                    python -c "
from nemo.collections.llm import peft
peft.merge_lora(base_path=\"/base\", adapter_path=\"/adapter\", output_path=\"/dst\")
"
                '
        fi
        ;;

    retinanet|dlrm_dcnv2|rgat|flux1)
        warn "Weight conversion is not defined for $WL_NAME."
        info "This workload uses a bespoke checkpoint format managed by its own scripts."
        info "See: $WL_DOC_URL"
        exit 0
        ;;

    *)
        die "Unknown workload: $WL_NAME"
        ;;
esac

say "Done. Output at: $DST"
