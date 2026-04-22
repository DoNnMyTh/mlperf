# Workload manifest — Llama 3.1 405B (NeMo, C4 preprocessed with Mixtral 8x22b tokenizer)
# Resumes from Meta's official HF 405B BF16 checkpoint.

WL_NAME="llama31_405b"
WL_DISPLAY="Llama 3.1 405B (NeMo)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/llama31_405b/implementations/theia_ngpu512_ngc25.09_nemo"
WL_HUB_REPO=""
WL_IMAGE_TAG_BASE="llama31_405b-pyt"
WL_IMAGE_TAG_VARIANTS=()

# Dataset
# Final layout: $DATADIR/405b/{c4-*.{bin,idx}, tokenizer/}
# Note: this benchmark uses the Mixtral 8x22B tokenizer rather than TikToken.
WL_DATASET_SUBDIR="405b"
WL_DATASET_SIZE_GB=300
WL_DATASET_MARKER_FILES=("c4-train.en_6_text_document.bin" "c4-train.en_7_text_document.bin")
WL_DATASET_MARKER_DIRS=("tokenizer")
WL_DOWNLOAD_SCRIPT="data_scripts/download.sh"
WL_DOWNLOAD_ENV="DATADIR=/data"
WL_DOWNLOAD_HOST_ENV="DATADIR=\$DATADIR"

# Mounts per config_mounts.sh (selected by MODEL_SIZE=405b inside configs):
#   $DATADIR/405b          -> /preproc_data
#   $DATADIR/405b/tokenizer -> /workspace/llm/nemo_tokenizer
# Checkpoint is a *separate* host path (LOAD_CHECKPOINTS_PATH) mounted to
# /load_checkpoints; the driver does not automate its staging (~1.5TB).
WL_PREPROC_HOST_SUBPATH="405b"
WL_PREPROC_MOUNT="/preproc_data"
WL_TOKENIZER_HOST_SUBPATH="405b/tokenizer"
WL_TOKENIZER_MOUNT="/workspace/llm/nemo_tokenizer"

WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="pretrain.py"
WL_CONTAINER_WORKDIR="/workspace/llm"

# Full 405B model does not fit on any single consumer GPU; even a 2-layer
# smoke mirrors the 8B OOM pattern. Provide synthetic-data smoke path that
# short-circuits checkpoint loading for infra plumbing verification only.
WL_SMOKE_SUPPORTED=1
WL_SMOKE_ENV=(
    DGXNGPU=1 DGXNNODES=1
    TENSOR_MODEL_PARALLEL=1 PIPELINE_MODEL_PARALLEL=1 CONTEXT_PARALLEL=1
    SEQ_PARALLEL=False INTERLEAVED_PIPELINE=0
    MICRO_BATCH_SIZE=1 MINIBS=1
    USE_SYNTHETIC_DATA=1
    FP8=False FP8_HYBRID=False FP8_PARAM_GATHER=False
    TP_COMM_OVERLAP=False MC_TP_OVERLAP_AG=False MC_TP_OVERLAP_RS=False
    OVERLAP_PARAM_GATHER=False OVERLAP_GRAD_REDUCE=False USE_DIST_OPTIMIZER=False
    VAL_CHECK_INTERVAL=999 WARMUP_STEPS=0 LOAD_CHECKPOINT=""
    BINDCMD=""
)
WL_SMOKE_PROMPTS=(
    "MAX_STEPS:3"
    "OVERWRITTEN_NUM_LAYERS:2"
)

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/theia_ngpu512_ngc25.09_nemo"
WL_DOCKERFILE_PATCH_FROM='NVTE_CUDA_ARCHS="100a;103a"'
WL_DOCKERFILE_PATCH_TO='NVTE_CUDA_ARCHS="89;100a;103a"'
