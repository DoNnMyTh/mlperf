# Workload manifest — DLRM DCNv2 (HugeCTR)
# Criteo 1TB Click Logs -> NumPy -> HugeCTR raw format.

WL_NAME="dlrm_dcnv2"
WL_DISPLAY="DLRM DCNv2 (HugeCTR, Criteo 1TB)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr"
WL_HUB_REPO=""
WL_IMAGE_TAG_BASE="recommendation-hugectr"
WL_IMAGE_TAG_VARIANTS=()

# Dataset: ~4 TB processed (raw train_data.bin + val_data.bin + intermediates).
# Raw Criteo download is ~1 TB; .gz unpack + NumPy + multi-hot + raw conversion
# requires ~8 TB total working space and up to 5 days of preprocessing.
WL_DATASET_SUBDIR="criteo_1tb_multihot_raw"
WL_DATASET_SIZE_GB=8000
WL_DATASET_MARKER_FILES=("train_data.bin" "val_data.bin")
WL_DATASET_MARKER_DIRS=()
# No single-shot download script exists; preprocessing is a documented chain
# (see upstream README). Driver surfaces this and refuses to auto-run.
WL_DOWNLOAD_SCRIPT=""
WL_DOWNLOAD_ENV=""
WL_DOWNLOAD_HOST_ENV=""

# Mounts — both /data and /data_val point at the same dir by default
WL_PREPROC_HOST_SUBPATH="criteo_1tb_multihot_raw"
WL_PREPROC_MOUNT="/data"
WL_TOKENIZER_HOST_SUBPATH=""
WL_TOKENIZER_MOUNT=""

WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="train.py"
WL_CONTAINER_WORKDIR="/workspace/dlrm"

# HugeCTR does not ship a single-GPU "mock" mode that satisfies the pipeline;
# skip smoke.
WL_SMOKE_SUPPORTED=0
WL_SMOKE_ENV=()
WL_SMOKE_PROMPTS=()

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr"
WL_DOCKERFILE_PATCH_FROM=""
WL_DOCKERFILE_PATCH_TO=""
