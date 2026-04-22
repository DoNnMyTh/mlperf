# Workload manifest — RetinaNet / SSD (PyTorch)
# OpenImages-v6 MLPerf subset; ResNeXt50_32x4d backbone.

WL_NAME="retinanet"
WL_DISPLAY="RetinaNet / SSD (PyTorch, OpenImages-v6)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/retinanet/implementations/tyche_ngpu8_ngc25.04_pytorch"
WL_HUB_REPO="donnmyth/mlperf-nvidia"
WL_IMAGE_TAG_BASE="single_stage_detector-pyt"
# Single pushed tag; empty-string variant means "no -<variant> suffix" and
# pulls donnmyth/mlperf-nvidia:single_stage_detector-pyt as-is.
WL_IMAGE_TAG_VARIANTS=("")

# Dataset — OpenImages-v6 MLPerf subset
# Expected layout: $DATADIR/{train,validation}/{data,labels,metadata}/...
# We treat $DATADIR/$WL_DATASET_SUBDIR as the root containing train/ and validation/.
WL_DATASET_SUBDIR="open-images-v6"
WL_DATASET_SIZE_GB=400
WL_DATASET_MARKER_FILES=("info.json" "train/labels/openimages-mlperf.json" "validation/labels/openimages-mlperf.json")
WL_DATASET_MARKER_DIRS=("train/data" "validation/data")
# Download requires fiftyone pip package and runs only usefully inside the
# workload container. We point the driver at the official downloader.
WL_DOWNLOAD_SCRIPT="public-scripts/download_openimages_mlperf.sh"
WL_DOWNLOAD_ENV="FIFTYONE_DATASET_ZOO_DIR=/data/open-images-v6"
WL_DOWNLOAD_HOST_ENV="FIFTYONE_DATASET_ZOO_DIR=\$DATADIR/open-images-v6"

# Mounts
WL_PREPROC_HOST_SUBPATH="open-images-v6"
WL_PREPROC_MOUNT="/datasets/open-images-v6"
# Pretrained backbone is a separate host path in the README; surface via tokenizer-slot
# so the driver's optional mount is used.
WL_TOKENIZER_HOST_SUBPATH="open-images-v6/torch-home"
WL_TOKENIZER_MOUNT="/root/.cache/torch"

# Configs
WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

# Launch
WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="train.py"
WL_CONTAINER_WORKDIR="/workspace/ssd"

# Synthetic-data smoke is supported by run_and_time.sh (USE_SYNTHETIC_DATA=1).
WL_SMOKE_SUPPORTED=1
WL_SMOKE_ENV=(
    DGXNGPU=1 DGXNNODES=1
    USE_SYNTHETIC_DATA=1
    WARMUP_STEPS=0
    BINDCMD=""
)
WL_SMOKE_PROMPTS=(
    "MAX_STEPS:3"
)

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/retinanet/implementations/tyche_ngpu8_ngc25.04_pytorch"
WL_DOCKERFILE_PATCH_FROM=""
WL_DOCKERFILE_PATCH_TO=""
