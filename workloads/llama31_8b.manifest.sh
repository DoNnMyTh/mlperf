# Workload manifest — Llama 3.1 8B (NeMo)
# Sourced by mlperf.sh after the user picks this workload.
# All WL_* variables form the contract between driver and workload.

WL_NAME="llama31_8b"
WL_DISPLAY="Llama 3.1 8B (NeMo)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/llama31_8b/implementations/nemo"
WL_HUB_REPO="donnmyth/mlperf-nvidia"
WL_IMAGE_TAG_BASE="llama31_8b-pyt"
# Optional published tag variants: <base>-blackwell , <base>-sm89
WL_IMAGE_TAG_VARIANTS=("blackwell" "sm89")

# Dataset
WL_DATASET_SUBDIR="8b"                 # $DATADIR/$WL_DATASET_SUBDIR is the root
WL_DATASET_SIZE_GB=100
WL_DATASET_MARKER_FILES=("c4-train.en_6_text_document.bin" "c4-train.en_6_text_document.idx")
WL_DATASET_MARKER_DIRS=("tokenizer")
WL_DOWNLOAD_SCRIPT="data_scripts/download_8b.sh"
WL_DOWNLOAD_ENV="DATADIR=/data/8b"     # how the script is invoked in-container
WL_DOWNLOAD_HOST_ENV="DATADIR=\$DATADIR/8b"  # bare-metal equivalent

# In-container mount points expected by run_and_time.sh / config_mounts.sh
WL_PREPROC_HOST_SUBPATH="8b"
WL_PREPROC_MOUNT="/preproc_data"
WL_TOKENIZER_HOST_SUBPATH="8b/tokenizer"
WL_TOKENIZER_MOUNT="/workspace/llm/nemo_tokenizer"

# Config discovery
WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

# Launch
WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="pretrain.py"
WL_CONTAINER_WORKDIR="/workspace/llm"

# Single-GPU smoke-test support
WL_SMOKE_SUPPORTED=1
WL_SMOKE_ENV=(
    DGXNGPU=1 DGXNNODES=1
    TENSOR_MODEL_PARALLEL=1 PIPELINE_MODEL_PARALLEL=1 CONTEXT_PARALLEL=1
    SEQ_PARALLEL=False INTERLEAVED_PIPELINE=0
    MICRO_BATCH_SIZE=1 MINIBS=1
    FP8=False FP8_HYBRID=False FP8_PARAM_GATHER=False
    TP_COMM_OVERLAP=False MC_TP_OVERLAP_AG=False MC_TP_OVERLAP_RS=False
    OVERLAP_PARAM_GATHER=False OVERLAP_GRAD_REDUCE=False USE_DIST_OPTIMIZER=False
    VAL_CHECK_INTERVAL=999 WARMUP_STEPS=0 LOAD_CHECKPOINT=""
    BINDCMD=""
)
# Extra smoke knobs asked from user (var=prompt pairs)
WL_SMOKE_PROMPTS=(
    "MAX_STEPS:3"
    "OVERWRITTEN_NUM_LAYERS:2"
)

# Documentation link shown in Step 1
WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_8b/implementations/nemo"

# Optional Dockerfile sm_89 patch (for non-Blackwell dev on this workload)
WL_DOCKERFILE_PATCH_FROM='NVTE_CUDA_ARCHS="100a;103a"'
WL_DOCKERFILE_PATCH_TO='NVTE_CUDA_ARCHS="89;100a;103a"'
