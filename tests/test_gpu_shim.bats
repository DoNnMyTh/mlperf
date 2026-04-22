#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/.." && pwd)"
    # shellcheck source=../lib/common.sh
    source "$REPO_ROOT/lib/common.sh"
    # Prepend fake nvidia-smi
    chmod +x "$REPO_ROOT/tests/fixtures/bin/nvidia-smi"
    export PATH="$REPO_ROOT/tests/fixtures/bin:$PATH"
    unset CUDA_VISIBLE_DEVICES SLURM_GPUS_ON_NODE
}

# detect_gpus lives in mlperf.sh. Extract it to a self-contained function
# here to avoid sourcing the whole interactive script.
detect_gpus() {
    GPU_TOTAL=0; GPU_LIST=""; GPU_NAMES=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPU_LIST="$CUDA_VISIBLE_DEVICES"
        GPU_TOTAL=$(awk -F',' '{print NF}' <<<"$GPU_LIST")
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)
        GPU_TOTAL="${#GPU_NAMES[@]}"
        (( GPU_TOTAL > 0 )) && GPU_LIST=$(seq -s, 0 $((GPU_TOTAL-1)))
    fi
}

@test "detect_gpus via fake nvidia-smi returns 4 GPUs" {
    detect_gpus
    [[ "$GPU_TOTAL" == "4" ]]
    [[ "$GPU_LIST" == "0,1,2,3" ]]
}

@test "detect_gpus honours CUDA_VISIBLE_DEVICES" {
    export CUDA_VISIBLE_DEVICES="1,3"
    detect_gpus
    [[ "$GPU_TOTAL" == "2" ]]
    [[ "$GPU_LIST" == "1,3" ]]
}

@test "fake nvidia-smi compute_cap returns 9.0 (Hopper)" {
    [[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)" == "9.0" ]]
}
