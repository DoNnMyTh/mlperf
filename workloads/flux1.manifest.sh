# Workload manifest — FLUX.1 text-to-image (NeMo, WebDataset/Energon)
# CC12M + COCO preprocessed shards; T5/CLIP empty encodings.

WL_NAME="flux1"
WL_DISPLAY="FLUX.1 text-to-image (NeMo)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/flux1/implementations/theia_ngpu16_ngc25.09_nemo"
WL_HUB_REPO="donnmyth/mlperf-nvidia"
WL_IMAGE_TAG_BASE="flux1-pyt"
WL_IMAGE_TAG_VARIANTS=("")

# Dataset — Energon WebDataset format after preprocessing.
# Final layout: $DATADIR/energon/{train/*.tar,val/*.tar,empty_encodings/}
WL_DATASET_SUBDIR="energon"
WL_DATASET_SIZE_GB=6000
WL_DATASET_MARKER_FILES=("train/index.txt" "empty_encodings/clip_empty.npy" "empty_encodings/t5_empty.npy")
WL_DATASET_MARKER_DIRS=("train" "val" "empty_encodings")
# Upstream provides a slurm "download.sub" wrapper that runs inside the
# container; the raw curl/energon-prepare chain is multi-step and interactive
# (prompts for duplicate keys and class selection). Leave unset so the driver
# surfaces the manual procedure.
WL_DOWNLOAD_SCRIPT=""
WL_DOWNLOAD_ENV=""
WL_DOWNLOAD_HOST_ENV=""

# Mounts: run_and_time.sh uses DATAROOT -> /dataset/energon
WL_PREPROC_HOST_SUBPATH="energon"
WL_PREPROC_MOUNT="/dataset/energon"
WL_TOKENIZER_HOST_SUBPATH=""
WL_TOKENIZER_MOUNT=""

WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="train.py"
WL_CONTAINER_WORKDIR="/workspace/flux"

# FLUX supports synthetic data via USE_SYNTHETIC_DATA=1 which short-circuits
# dataset mounts and enables mock-encoding generation.
WL_SMOKE_SUPPORTED=1
WL_SMOKE_ENV=(
    DGXNGPU=1 DGXNNODES=1
    USE_SYNTHETIC_DATA=1
    BINDCMD=""
)
WL_SMOKE_PROMPTS=(
    "MAX_STEPS:3"
)

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/flux1/implementations/theia_ngpu16_ngc25.09_nemo"
WL_DOCKERFILE_PATCH_FROM='NVTE_CUDA_ARCHS="100a;103a"'
WL_DOCKERFILE_PATCH_TO='NVTE_CUDA_ARCHS="89;90;100a;103a"'
