# Workload manifest — Llama 2 70B LoRA fine-tune (NeMo)
# GovReport dataset + Llama-2-70B checkpoint in NeMo distributed format.

WL_NAME="llama2_70b_lora"
WL_DISPLAY="Llama 2 70B LoRA fine-tune (NeMo, GovReport)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo"
WL_HUB_REPO="donnmyth/mlperf-nvidia"
WL_IMAGE_TAG_BASE="llama2_70b_lora-pyt"
WL_IMAGE_TAG_VARIANTS=("")

# Dataset layout after preprocessing:
#   $DATADIR/gov_report/{train.npy, validation.npy}
#   $DATADIR/model/{context/, weights/}
# Driver models this via marker files on the gov_report side and a second
# mount slot for the model checkpoint.
WL_DATASET_SUBDIR="gov_report"
WL_DATASET_SIZE_GB=300
WL_DATASET_MARKER_FILES=("train.npy" "validation.npy")
WL_DATASET_MARKER_DIRS=()
# Downloader is two Python scripts; they run inside the container. The driver
# can invoke them in sequence via a tiny wrapper.
WL_DOWNLOAD_SCRIPT="scripts/download_all.sh"
WL_DOWNLOAD_ENV="DATA_DIR=/data"
WL_DOWNLOAD_HOST_ENV="DATA_DIR=\$DATADIR"

# Container mounts per config_mounts.sh:
#   DATADIR -> /data (ro)
#   MODEL   -> /ckpt (ro)
WL_PREPROC_HOST_SUBPATH="gov_report"
WL_PREPROC_MOUNT="/data"
WL_TOKENIZER_HOST_SUBPATH="model"
WL_TOKENIZER_MOUNT="/ckpt"

WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="train.py"
WL_CONTAINER_WORKDIR="/workspace/ft-llm"

# LoRA fine-tune loads the full 70B checkpoint (~140GB in bf16) even with low-rank
# adapters — cannot meaningfully run single-GPU smoke.
WL_SMOKE_SUPPORTED=0
WL_SMOKE_ENV=()
WL_SMOKE_PROMPTS=()

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo"
WL_DOCKERFILE_PATCH_FROM='NVTE_CUDA_ARCHS="100a;103a"'
WL_DOCKERFILE_PATCH_TO='NVTE_CUDA_ARCHS="89;90;100a;103a"'
