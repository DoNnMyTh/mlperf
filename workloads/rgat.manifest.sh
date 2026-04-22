# Workload manifest — R-GAT (DGL, IGBH-Full graph)
# Heterogeneous GNN on the IGB-Heterogeneous graph (full split).
# Requires two data mounts: features (DATA_DIR) and graph structures (GRAPH_DIR).

WL_NAME="rgat"
WL_DISPLAY="R-GAT (DGL, IGBH-Full graph)"
WL_IMPL_SUBDIR="NVIDIA/benchmarks/rgat/implementations/tyche_ngpu8_ngc25.03_dgl"
WL_HUB_REPO=""
WL_IMAGE_TAG_BASE="graph_neural_network-dgl"
WL_IMAGE_TAG_VARIANTS=()

# Dataset
# After preprocessing the final layout is:
#   $DATADIR/converted/{author_node_feat.bin, concat_features.bin, ...}   # DATA_DIR_FLOAT8
#   $DATADIR/graph/{paper__cites__..., train_idx.pt, val_idx.pt, ...}    # GRAPH_DIR
# Raw IGBH download is ~2 TB; preprocessing expands to ~1.3 TB features + ~200 GB graph.
WL_DATASET_SUBDIR="converted"
WL_DATASET_SIZE_GB=6000
WL_DATASET_MARKER_FILES=("concat_features.bin" "config.yml")
WL_DATASET_MARKER_DIRS=()
# The downloader is a multi-step chain run inside the container; surface it
# but do not auto-invoke due to 2-3 day runtime.
WL_DOWNLOAD_SCRIPT="utility/download_igbh_full.sh"
WL_DOWNLOAD_ENV=""
WL_DOWNLOAD_HOST_ENV=""

# Mounts — use the manifest's two mount slots for DATA_DIR and GRAPH_DIR
WL_PREPROC_HOST_SUBPATH="converted"
WL_PREPROC_MOUNT="/data"
WL_TOKENIZER_HOST_SUBPATH="graph"
WL_TOKENIZER_MOUNT="/graph"

WL_CONFIG_GLOB="config_*.sh"
WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'

WL_ENTRY="run_and_time.sh"
WL_PRETRAIN_PY="train.py"
WL_CONTAINER_WORKDIR="/workspace/gnn"

# Smoke run is non-trivial because DGL requires graph loading; skip.
WL_SMOKE_SUPPORTED=0
WL_SMOKE_ENV=()
WL_SMOKE_PROMPTS=()

WL_DOC_URL="https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/rgat/implementations/tyche_ngpu8_ngc25.03_dgl"
WL_DOCKERFILE_PATCH_FROM=""
WL_DOCKERFILE_PATCH_TO=""
