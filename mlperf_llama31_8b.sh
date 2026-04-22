#!/usr/bin/env bash
# Interactive runner for MLPerf Llama 3.1 8B NeMo benchmark (NVIDIA).
# Supports: docker / enroot sqsh / bare-metal   x   sbatch / srun / local / multi-node / prepare-only
# Handles login nodes, cluster + workstation + single-node cases.

set -u
set -o pipefail

# ====================================================================
# bash version + TTY checks (edge-case #2, #3)
# ====================================================================
if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: Bash >= 4 required. Current: $BASH_VERSION" >&2
    echo "  macOS: 'brew install bash' and run with /usr/local/bin/bash or /opt/homebrew/bin/bash" >&2
    exit 1
fi
if [[ ! -t 0 ]]; then
    echo "ERROR: non-interactive stdin (piped/CI). This script requires TTY prompts." >&2
    echo "  Run manually in a terminal." >&2
    exit 1
fi

# ====================================================================
# configuration
# ====================================================================
REPO_URL="https://github.com/mlcommons/training_results_v5.1.git"
REPO_SUBDIR="NVIDIA/benchmarks/llama31_8b/implementations/nemo"
HUB_REPO="donnmyth/mlperf-nvidia"
IMG_TAG_BASE="llama31_8b-pyt"
SCRIPT_VERSION="2.0"

IMAGE=""; SQSH=""; CONT_REF=""; CFG_FILE=""; IS_CUSTOM=0
MAX_STEPS=3; LAYERS=2; NGPU=1
DGXNNODES=1; DGXNGPU=1; WALLTIME="30"
NEED_DOCKER=0; NEED_ENROOT=0; NEED_BARE=0
GPU_TOTAL=0; GPU_LIST=""; declare -a GPU_NAMES=()
declare -a CLEANUP_CONTAINERS=()
RUN_ON_LOGIN_NODE=0

# ====================================================================
# ui helpers
# ====================================================================
say()  { printf "\n==> %s\n" "$*"; }
info() { printf "    %s\n" "$*"; }
warn() { printf "WARN: %s\n" "$*" >&2; }
err()  { printf "ERROR: %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

cleanup() {
    local c
    for c in "${CLEANUP_CONTAINERS[@]:-}"; do
        [[ -n "$c" ]] && docker kill "$c" >/dev/null 2>&1 || true
    done
}
on_abort() { err "Aborted by signal."; cleanup; exit 130; }
on_exit()  { cleanup; }
trap on_abort INT TERM
trap on_exit  EXIT

ask()     { local p="$1" d="${2-}" v=""
            if [[ -n "$d" ]]; then read -r -p "$p [$d]: " v; echo "${v:-$d}"
            else                   read -r -p "$p: "        v; echo "$v"; fi
          }
ask_req() { local p="$1" v=""
            while :; do read -r -p "$p: " v; [[ -n "$v" ]] && { echo "$v"; return; }
                         err "value required"; done
          }
yesno()   { local p="$1" d="${2-y}" v=""
            while :; do read -r -p "$p (y/n) [$d]: " v; v="${v:-$d}"
                case "$v" in [Yy]|[Yy][Ee][Ss]) return 0;;
                             [Nn]|[Nn][Oo])      return 1;;
                             *) err "Answer y or n";; esac
            done
          }
pick()    { local p="$1"; shift
            local i=1; for o in "$@"; do printf "  [%d] %s\n" "$i" "$o" >&2; i=$((i+1)); done
            local v=""
            while :; do read -r -p "$p [1]: " v; v="${v:-1}"
                [[ "$v" =~ ^[0-9]+$ ]] && (( v>=1 && v<=$# )) && { echo "$v"; return; }
                err "Enter 1..$#"
            done
          }

# Validate user-supplied path has no space/special chars that break mounts/args.
# (edge-case #4)
validate_path() {
    local p="$1" label="$2"
    if [[ "$p" =~ [[:space:]] ]]; then
        die "$label path must not contain spaces: '$p'"
    fi
    if [[ "$p" =~ [,\;\|\&\$\`\"\'\(\)] ]]; then
        die "$label path has shell-special chars: '$p'"
    fi
}

# ====================================================================
# platform detect
# ====================================================================
OS="$(uname -s 2>/dev/null || echo unknown)"
case "$OS" in
    Linux*)              PLATFORM=linux   ;;
    Darwin*)             PLATFORM=mac     ;;
    MINGW*|MSYS*|CYGWIN*)PLATFORM=windows ;;
    *)                   PLATFORM=unknown ;;
esac
PKG=""
if [[ "$PLATFORM" == "linux" ]]; then
    for m in apt-get dnf yum pacman zypper apk; do
        command -v "$m" >/dev/null 2>&1 && { PKG="$m"; break; }
    done
fi
SUDO=""
[[ ${EUID:-$(id -u)} -ne 0 ]] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
if [[ "$PLATFORM" == "windows" ]]; then
    export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
fi

# ====================================================================
# install helpers (edge-case #26: confirm sudo escalation)
# ====================================================================
pkg_install() {
    local p="$1"
    [[ -z "$PKG" ]] && { err "No package manager; install $p manually."; return 1; }
    if [[ -n "$SUDO" ]]; then
        yesno "Run '$SUDO $PKG install $p' (requires admin password)?" y || return 1
    fi
    case "$PKG" in
        apt-get) $SUDO apt-get update && $SUDO apt-get install -y "$p" ;;
        dnf|yum) $SUDO "$PKG" install -y "$p" ;;
        pacman)  $SUDO pacman -S --noconfirm "$p" ;;
        zypper)  $SUDO zypper install -y "$p" ;;
        apk)     $SUDO apk add "$p" ;;
    esac
}

guide_install() {
    local tool="$1"
    echo "  How to install '$tool' on $PLATFORM:"
    case "$PLATFORM-$tool" in
        windows-git)    echo "    https://git-scm.com/download/win" ;;
        windows-docker) echo "    https://docs.docker.com/desktop/install/windows-install/" ;;
        windows-nvidia-container-toolkit)
                        echo "    WSL2 GPU: https://docs.nvidia.com/cuda/wsl-user-guide/" ;;
        mac-git)        echo "    xcode-select --install   or   brew install git" ;;
        mac-docker)     echo "    https://docs.docker.com/desktop/install/mac-install/" ;;
        linux-*)        echo "    Use package manager, or vendor docs." ;;
        *)              echo "    See official docs for '$tool'." ;;
    esac
}

require_tool() {
    local tool="$1"
    if command -v "$tool" >/dev/null 2>&1; then
        info "$tool: $(command -v "$tool")"
        return 0
    fi
    err "$tool not found in PATH."
    if [[ "$PLATFORM" == "linux" && -n "$PKG" ]] && yesno "Attempt install via $PKG?" y; then
        pkg_install "$tool" && command -v "$tool" >/dev/null 2>&1 && { info "$tool installed"; return 0; }
        err "Auto-install failed."
    fi
    guide_install "$tool"
    die "Cannot continue without $tool."
}

wait_for_docker() {
    if ! docker info >/dev/null 2>&1; then
        err "Docker daemon unreachable."
        guide_install docker
        yesno "Wait for Docker to start and retry?" y || die "Docker required."
        local t=0
        while ! docker info >/dev/null 2>&1; do
            sleep 2; printf "."; t=$((t+2))
            (( t > 180 )) && die "Timed out waiting for Docker."
        done
        echo; info "Docker is up."
    fi
    info "docker daemon: $(docker info --format '{{.ServerVersion}}' 2>/dev/null)"
}

# edge-case #8: handle private registry 401
docker_pull_with_auth() {
    local img="$1"
    if docker pull "$img" 2>&1 | tee /tmp/.pull.$$ | grep -qE "unauthorized|denied|authentication required|401"; then
        rm -f /tmp/.pull.$$
        warn "Pull failed — looks like a private registry."
        local host; host="$(awk -F'/' '{print $1}' <<<"$img")"
        [[ "$host" != *.* ]] && host="docker.io"
        yesno "Run 'docker login $host' now?" y || die "Cannot pull without auth."
        docker login "$host" || die "docker login failed"
        docker pull "$img" || die "pull still failed"
    else
        rm -f /tmp/.pull.$$
    fi
}

enroot_import_with_auth() {
    local out="$1" ref="$2"
    local dir; dir="$(dirname "$out")"
    # edge-case #24: writable target dir
    [[ -w "$dir" ]] || die "No write permission on $dir for enroot sqsh output."
    if ! enroot import -o "$out" "$ref" 2>/tmp/.enroot.$$; then
        if grep -qE "401|unauthorized|credential" /tmp/.enroot.$$ 2>/dev/null; then
            rm -f /tmp/.enroot.$$
            warn "enroot import auth error."
            info "Configure creds at ~/.config/enroot/.credentials"
            info "Format: 'machine auth.docker.io login <user> password <token>'"
            die "Set creds and retry."
        fi
        rm -f /tmp/.enroot.$$
        die "enroot import failed"
    fi
    rm -f /tmp/.enroot.$$
}

check_nvidia() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu; gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        [[ -n "$gpu" ]] && info "GPU: $gpu"
    else
        warn "nvidia-smi not on host PATH."
        yesno "Continue (container/WSL2 may still see GPU)?" y || die "Aborted."
    fi
}

# GPU arch code (sm_XX) of first GPU; 0 if unknown.
gpu_arch_code() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        [[ -n "$cc" ]] && { echo "$cc"; return; }
    fi
    echo 0
}

# edge-case #1 + #37: login-node friendly GPU detection
detect_gpus() {
    GPU_TOTAL=0; GPU_LIST=""; GPU_NAMES=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPU_LIST="$CUDA_VISIBLE_DEVICES"
        GPU_TOTAL=$(awk -F',' '{print NF}' <<<"$GPU_LIST")
        info "CUDA_VISIBLE_DEVICES=$GPU_LIST  ($GPU_TOTAL GPUs)"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)
        GPU_TOTAL="${#GPU_NAMES[@]}"
        if (( GPU_TOTAL > 0 )); then
            GPU_LIST=$(seq -s, 0 $((GPU_TOTAL-1)))
            info "Detected $GPU_TOTAL GPUs:"
            local line; for line in "${GPU_NAMES[@]}"; do info "  $line"; done
            return
        fi
    fi
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        GPU_TOTAL="$SLURM_GPUS_ON_NODE"
        GPU_LIST=$(seq -s, 0 $((GPU_TOTAL-1)))
        info "Slurm SLURM_GPUS_ON_NODE=$GPU_TOTAL"
    fi
}

# Ask NGPU from total. If zero (login node), accept manual input non-fatally.
choose_gpus() {
    detect_gpus
    if (( GPU_TOTAL == 0 )); then
        warn "No GPU visible here (login node?)."
        if yesno "Continue anyway (will be set on allocated nodes)?" y; then
            RUN_ON_LOGIN_NODE=1
            NGPU="$(ask 'GPUs per node (per config/target cluster)' "${DGXNGPU:-8}")"
            return
        fi
        die "No GPUs and user declined to continue."
    fi
    NGPU="$(ask "GPUs to use (1..$GPU_TOTAL)" "$GPU_TOTAL")"
    [[ "$NGPU" =~ ^[0-9]+$ ]] && (( NGPU >= 1 && NGPU <= GPU_TOTAL )) || die "Invalid NGPU=$NGPU"
    if (( NGPU < GPU_TOTAL )); then
        local default_sel; default_sel=$(seq -s, 0 $((NGPU-1)))
        local subset; subset="$(ask "GPU indices (comma-separated)" "$default_sel")"
        export CUDA_VISIBLE_DEVICES="$subset"
        info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
}

# edge-case #7: disk space (GB) free on filesystem containing $1
free_gb() {
    local path="$1" gb
    if [[ "$PLATFORM" == "mac" ]]; then
        gb=$(df -g "$path" 2>/dev/null | awk 'NR==2 {print $4}')
    else
        gb=$(df -BG "$path" 2>/dev/null | awk 'NR==2 {gsub(/G$/,"",$4); print $4}')
    fi
    # strip any remaining non-digits
    gb="${gb//[^0-9]/}"
    echo "${gb:-0}"
}

need_space_gb() {
    local path="$1" need="$2" label="$3"
    local have; have=$(free_gb "$path")
    info "$label: ${have}G free (need ~${need}G)"
    if (( have < need )); then
        warn "Low disk at $path (${have}G < ${need}G)."
        yesno "Continue anyway?" n || die "Free up space and retry."
    fi
}

# edge-case #6: un-nest E:/.../8b/8b/ that earlier duplicate runs can produce
fix_nested_dataset() {
    local root="$1"
    if [[ -d "$root/8b/8b" && -f "$root/8b/8b/c4-train.en_6_text_document.bin" && ! -f "$root/8b/c4-train.en_6_text_document.bin" ]]; then
        warn "Detected nested $root/8b/8b/ (from duplicate cleanup)."
        if yesno "Un-nest by moving $root/8b/8b/* -> $root/8b/ ?" y; then
            # Handle conflicts: remove empty/duplicate siblings at outer level.
            ( cd "$root/8b" || exit 1
              shopt -s dotglob nullglob
              local f
              for f in 8b/*; do
                  local name; name="$(basename "$f")"
                  if [[ -e "$name" && ! "$name" == "8b" ]]; then
                      # outer has same name; prefer the inner copy (more recent).
                      rm -rf "$name"
                  fi
                  mv -- "$f" .
              done
              rmdir 8b
            ) || die "Un-nest failed."
            info "Un-nested."
        fi
    fi
}

# edge-case #5: md5 verification option (uses md5 files in dataset)
verify_dataset_md5() {
    local dir="$1"
    command -v md5sum >/dev/null 2>&1 || { warn "md5sum missing; skipping verify."; return; }
    local f
    for f in "$dir"/*.md5; do
        [[ -f "$f" ]] || continue
        info "Checking $(basename "$f")..."
        ( cd "$dir" && md5sum --check --quiet "$(basename "$f")" ) \
            && info "  OK" || { err "MD5 mismatch in $f"; return 1; }
    done
}

# ====================================================================
# banner
# ====================================================================
cat <<BANNER

╔══════════════════════════════════════════════════════════════╗
║ MLPerf Llama 3.1 8B — interactive runner    v$SCRIPT_VERSION           ║
║ platform: $PLATFORM   pkg: ${PKG:-none}   bash: ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}
╚══════════════════════════════════════════════════════════════╝
BANNER

# ====================================================================
# step 0: preflight
# ====================================================================
say "Step 0: preflight"
require_tool git

# ====================================================================
# step 1: repo
# ====================================================================
say "Step 1: repository"
if yesno "Already have training_results_v5.1 cloned?" n; then
    REPO_DIR="$(ask_req 'Absolute path to existing repo')"
    validate_path "$REPO_DIR" "repo"
    [[ -d "$REPO_DIR" ]] || die "Path not found: $REPO_DIR"
else
    REPO_DIR="$(ask 'Where to clone' "$PWD/training_results_v5.1")"
    validate_path "$REPO_DIR" "repo"
    if [[ -e "$REPO_DIR" ]]; then
        yesno "$REPO_DIR exists. Reuse?" y || die "Pick a different path."
    else
        if yesno "Shallow clone (depth=1, faster)?" y; then
            git clone --depth 1 "$REPO_URL" "$REPO_DIR" || die "clone failed"
        else
            git clone "$REPO_URL" "$REPO_DIR" || die "clone failed"
        fi
    fi
fi
NEMO_DIR="$REPO_DIR/$REPO_SUBDIR"
[[ -f "$NEMO_DIR/Dockerfile" ]] || die "Dockerfile missing at $NEMO_DIR"
cd "$NEMO_DIR"
info "CWD: $PWD"

# ====================================================================
# step 2: container source
# ====================================================================
say "Step 2: container source"
HAS_ENROOT=0; command -v enroot >/dev/null 2>&1 && HAS_ENROOT=1

# edge-case #17: bare-metal on Windows not viable
BARE_OK=1
if [[ "$PLATFORM" == "windows" ]]; then
    BARE_OK=0
    info "Windows host: bare-metal disabled (no nvcc/TE build chain)."
fi

OPTS=(
    "docker: build locally (Dockerfile)"
    "docker: pull $HUB_REPO:$IMG_TAG_BASE-blackwell   (sm_100/103 only)"
    "docker: pull $HUB_REPO:$IMG_TAG_BASE-sm89        (adds sm_89 for 4080/4090)"
)
(( BARE_OK == 1 )) && OPTS+=("none — bare-metal (skip container, use host Python)")
if (( HAS_ENROOT == 1 )); then
    OPTS+=("enroot: import from registry to .sqsh (no docker)")
    OPTS+=("enroot: use existing .sqsh file")
else
    info "enroot not detected — enroot options hidden."
fi
sel=$(pick "Container source" "${OPTS[@]}")
CHOICE="${OPTS[$((sel-1))]}"
info "Selected: $CHOICE"

case "$CHOICE" in
    docker:*)  NEED_DOCKER=1 ;;
    enroot:*)  NEED_ENROOT=1 ;;
    none*)     NEED_BARE=1   ;;
esac

if (( NEED_DOCKER == 1 )); then
    require_tool docker
    wait_for_docker
    check_nvidia
    # edge-case #18: arch mismatch warning
    _local_arch=$(gpu_arch_code)
    if [[ "$CHOICE" == *"-blackwell"* ]] && (( _local_arch > 0 && _local_arch < 100 )); then
        warn "Pulled image supports sm_100/103 only, but detected GPU sm_$_local_arch."
        yesno "Continue anyway (kernels will fail at runtime)?" n || die "Aborted."
    fi
    unset _local_arch
fi
if (( NEED_ENROOT == 1 )); then
    require_tool enroot
    check_nvidia
fi

case "$sel" in
    1)  IMAGE="$(ask 'Local image name:tag' "mlperf-nvidia:$IMG_TAG_BASE")"
        if yesno "Patch Dockerfile NVTE_CUDA_ARCHS to add 89 (RTX 40xx/Ada)?" n; then
            if grep -q 'NVTE_CUDA_ARCHS="100a;103a"' Dockerfile; then
                sed -i 's/NVTE_CUDA_ARCHS="100a;103a"/NVTE_CUDA_ARCHS="89;100a;103a"/' Dockerfile  # edge-case #12: no .bak
                info "Patched: $(grep NVTE_CUDA_ARCHS Dockerfile)"
            elif grep -q 'NVTE_CUDA_ARCHS="89;100a;103a"' Dockerfile; then
                info "Already patched."
            else
                warn "Pattern not found; skipping patch."
            fi
        fi
        # edge-case #7: disk space for build
        need_space_gb "$(dirname "$NEMO_DIR")" 80 "build dir"
        yesno "Run 'docker build' now?" y || die "Cannot run without an image."
        docker build -t "$IMAGE" . || die "build failed"
        ;;
    2)  IMAGE="$HUB_REPO:$IMG_TAG_BASE-blackwell"; docker_pull_with_auth "$IMAGE" ;;
    3)  IMAGE="$HUB_REPO:$IMG_TAG_BASE-sm89";      docker_pull_with_auth "$IMAGE" ;;
    *)
        case "$CHOICE" in
            none*)
                : ;;  # bare-metal
            "enroot: import"*)
                REG="$(ask 'Registry ref' "docker://$HUB_REPO:$IMG_TAG_BASE-sm89")"
                SQSH_OUT="$(ask 'Output .sqsh path' "$PWD/mlperf-nvidia_${IMG_TAG_BASE}.sqsh")"
                validate_path "$SQSH_OUT" "sqsh"
                yesno "Run: enroot import -o $SQSH_OUT $REG ?" y || die "Aborted."
                enroot_import_with_auth "$SQSH_OUT" "$REG"
                SQSH="$SQSH_OUT"
                ;;
            "enroot: use existing"*)
                SQSH="$(ask_req 'Absolute path to existing .sqsh file')"
                validate_path "$SQSH" "sqsh"
                [[ -f "$SQSH" ]] || die "sqsh not found: $SQSH"
                ;;
            *) die "Unhandled choice: $CHOICE" ;;
        esac ;;
esac

# edge-case #14: pyxis ref format — let user pick
if [[ -n "$IMAGE" ]]; then
    info "Pyxis container-image format:"
    info "  A) ${IMAGE//\//+}      (hub+name:tag — classic pyxis)"
    info "  B) docker://$IMAGE     (explicit schema, newer pyxis/enroot)"
    fmt=$(pick "Pyxis ref format" "${IMAGE//\//+}" "docker://$IMAGE" "skip (not using pyxis)")
    case "$fmt" in
        1) CONT_REF="${IMAGE//\//+}" ;;
        2) CONT_REF="docker://$IMAGE" ;;
        3) CONT_REF="" ;;
    esac
elif [[ -n "$SQSH" ]]; then
    CONT_REF="$SQSH"
fi
[[ -n "$CONT_REF" ]] && info "CONT (pyxis/enroot): $CONT_REF"
[[ -z "$IMAGE" && -z "$SQSH" ]] && info "No container — bare-metal mode."

# ====================================================================
# step 3: dataset
# ====================================================================
say "Step 3: dataset"
DATADIR="$(ask_req 'Absolute DATADIR (will contain 8b/)')"
validate_path "$DATADIR" "DATADIR"
yesno "Create $DATADIR if missing?" y && mkdir -p "$DATADIR"
export DATADIR

fix_nested_dataset "$DATADIR"  # edge-case #6

dataset_files_ok() {
    [[ -f "$DATADIR/8b/c4-train.en_6_text_document.bin" \
    && -f "$DATADIR/8b/c4-train.en_6_text_document.idx" \
    && -d "$DATADIR/8b/tokenizer" ]]
}

DO_DL=1
if dataset_files_ok; then
    if yesno "Dataset present at $DATADIR/8b. Skip download?" y; then
        DO_DL=0
        # edge-case #5: offer md5 verify
        yesno "Verify md5 checksums now?" n && verify_dataset_md5 "$DATADIR/8b"
    fi
fi

if (( DO_DL == 1 )); then
    need_space_gb "$DATADIR" 100 "DATADIR"
    if yesno "Download dataset now? (~80 GB, hours-long)" n; then
        if   [[ -n "$IMAGE" ]]; then
            docker run --rm --network=host \
                -v "$DATADIR:/data" -e DATADIR=/data/8b \
                "$IMAGE" bash data_scripts/download_8b.sh || die "download failed"
        elif [[ -n "$SQSH" ]]; then
            enroot start --mount "$DATADIR:/data" --env "DATADIR=/data/8b" \
                "$SQSH" bash /workspace/llm/data_scripts/download_8b.sh || die "download failed"
        else
            require_tool curl
            require_tool wget
            ( cd "$NEMO_DIR" && DATADIR="$DATADIR/8b" bash data_scripts/download_8b.sh ) \
                || die "download failed"
        fi
        fix_nested_dataset "$DATADIR"
    else
        warn "Skipped. Expected layout:"
        info "  $DATADIR/8b/{c4-*.bin,c4-*.idx,tokenizer/,LICENSE.txt,NOTICE.txt}"
        yesno "Continue with missing dataset?" n || die "Aborted."
    fi
fi

# ====================================================================
# step 4: config
# ====================================================================
say "Step 4: config"
mapfile -t CONFIGS < <(ls config_*.sh 2>/dev/null | grep -Ev '^config_common' || true)
(( ${#CONFIGS[@]} > 0 )) || die "No config_*.sh found in $PWD"

labels=("${CONFIGS[@]}")
labels+=("custom single-GPU smoke (TP=PP=CP=1, 2 layers, few steps; NOT compliant)")
sel=$(pick "Pick a config" "${labels[@]}")
if (( sel == ${#labels[@]} )); then
    IS_CUSTOM=1
    CFG_FILE=""
else
    CFG_FILE="${CONFIGS[$((sel-1))]}"
fi

# ====================================================================
# step 5: runtime params
# ====================================================================
say "Step 5: runtime parameters"
LOGDIR="$(ask_req 'Absolute LOGDIR (outputs)')"
validate_path "$LOGDIR" "LOGDIR"
yesno "Create $LOGDIR if missing?" y && mkdir -p "$LOGDIR"

# edge-case #10: LOGDIR non-empty
if [[ -d "$LOGDIR" ]] && [[ -n "$(ls -A "$LOGDIR" 2>/dev/null)" ]]; then
    warn "LOGDIR $LOGDIR not empty."
    if yesno "Use timestamped sub-dir to avoid clobber?" y; then
        LOGDIR="$LOGDIR/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$LOGDIR"
        info "New LOGDIR: $LOGDIR"
    fi
fi
export LOGDIR

SEED="$(ask 'SEED' 42)"

if (( IS_CUSTOM == 1 )); then
    MAX_STEPS="$(ask 'MAX_STEPS' 3)"
    LAYERS="$(ask 'OVERWRITTEN_NUM_LAYERS' 2)"
    choose_gpus
    if (( NGPU > 1 )); then
        warn "Custom smoke is 1-GPU only; overriding NGPU=1"
        NGPU=1
        # edge-case #11: guard unset var
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
    fi
else
    # edge-case #9: set +u around config source (MLPerf configs ref many optional vars)
    set +u
    # shellcheck disable=SC1090
    source "$CFG_FILE"
    set -u
    info "Config: DGXNNODES=${DGXNNODES:-?} DGXNGPU=${DGXNGPU:-?} WALLTIME=${WALLTIME:-?}"
    choose_gpus
fi

# edge-case #25: explain WALLTIME units
info "Note: WALLTIME uses MINUTES for docker/bare paths; HH:MM:SS for sbatch/srun."

# ====================================================================
# step 6: launcher selection
# ====================================================================
say "Step 6: launch method"
HAS_SBATCH=0; command -v sbatch >/dev/null 2>&1 && HAS_SBATCH=1
HAS_SRUN=0;   command -v srun   >/dev/null 2>&1 && HAS_SRUN=1
info "Detected: sbatch=$HAS_SBATCH srun=$HAS_SRUN enroot=$HAS_ENROOT docker=$NEED_DOCKER is_custom=$IS_CUSTOM bare_ok=$BARE_OK login_node=$RUN_ON_LOGIN_NODE"

OPTS=(); KEYS=()
add_opt(){ OPTS+=("$1"); KEYS+=("$2"); }

if (( HAS_SBATCH == 1 && IS_CUSTOM == 0 )) && [[ -n "$CONT_REF" ]]; then
    add_opt "sbatch run.sub  (MLPerf native, Slurm+Pyxis+Enroot, containerized)" "sbatch"
fi
if (( HAS_SRUN == 1 && HAS_ENROOT == 1 && IS_CUSTOM == 0 )) && [[ -n "$CONT_REF" ]]; then
    add_opt "srun + Pyxis/Enroot (interactive, containerized)" "srun"
fi
if (( HAS_SBATCH == 1 && IS_CUSTOM == 0 && BARE_OK == 1 )); then
    add_opt "sbatch bare-metal (Slurm, no container, host Python deps)" "sbatch_bare"
fi
if (( HAS_SRUN == 1 && IS_CUSTOM == 0 && BARE_OK == 1 )); then
    add_opt "srun bare-metal (interactive Slurm, no container)" "srun_bare"
fi
if (( RUN_ON_LOGIN_NODE == 0 )); then
    [[ -n "$IMAGE" ]] && add_opt "docker run (single-node; NOT MLPerf-compliant)" "docker"
    (( BARE_OK == 1 )) && add_opt "bare-metal torchrun (single-node, host Python)" "bare"
    if (( IS_CUSTOM == 0 && BARE_OK == 1 )); then
        add_opt "bare-metal torchrun multi-node (no Slurm; run on each node)" "bare_multi"
    fi
fi

if (( IS_CUSTOM == 1 )); then
    OPTS=(); KEYS=()
    [[ -n "$IMAGE" && $RUN_ON_LOGIN_NODE -eq 0 ]] && add_opt "docker smoke (single-GPU)" "smoke"
    (( BARE_OK == 1 && RUN_ON_LOGIN_NODE == 0 )) && add_opt "bare-metal smoke (single-GPU)" "smoke_bare"
fi

add_opt "prepare-only (stop here; print next command)" "prepare"

(( ${#OPTS[@]} > 0 )) || die "No launch method available for your selections."
sel=$(pick "Choose launcher" "${OPTS[@]}")
METHOD="${KEYS[$((sel-1))]}"
yesno "Proceed with '$METHOD'?" y || die "Aborted."

# ====================================================================
# dispatch helpers
# ====================================================================
# edge-case #23: pick a free MASTER_PORT
random_port() {
    local p
    while :; do
        p=$(( (RANDOM % 20000) + 20000 ))
        if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then
            echo $p; return
        fi
        exec 3<&- 3>&- 2>/dev/null || true
    done
}
DEFAULT_MPORT=$(random_port)

docker_common_args=(
    --rm --gpus all --ipc=host --shm-size=16g --network=host
    --ulimit memlock=-1 --ulimit stack=67108864
    -v "$DATADIR/8b:/preproc_data:ro"
    -v "$DATADIR/8b/tokenizer:/workspace/llm/nemo_tokenizer:ro"
    -v "$LOGDIR:/results"
    -e SEED="$SEED" -e WALLTIME="$WALLTIME"
    -e RANK=0 -e LOCAL_RANK=0 -e WORLD_SIZE=1 -e LOCAL_WORLD_SIZE=1
    -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT="$DEFAULT_MPORT"
    -e SLURM_JOB_ID=local -e SLURM_PROCID=0 -e SLURM_LOCALID=0
)
# Propagate CUDA_VISIBLE_DEVICES if user picked a subset (edge-case #41)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    docker_common_args+=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi

smoke_env=(
    DGXNGPU=1 DGXNNODES=1
    TENSOR_MODEL_PARALLEL=1 PIPELINE_MODEL_PARALLEL=1 CONTEXT_PARALLEL=1
    SEQ_PARALLEL=False INTERLEAVED_PIPELINE=0
    MICRO_BATCH_SIZE=1 MINIBS=1
    OVERWRITTEN_NUM_LAYERS="$LAYERS"
    FP8=False FP8_HYBRID=False FP8_PARAM_GATHER=False
    TP_COMM_OVERLAP=False MC_TP_OVERLAP_AG=False MC_TP_OVERLAP_RS=False
    OVERLAP_PARAM_GATHER=False OVERLAP_GRAD_REDUCE=False USE_DIST_OPTIMIZER=False
    MAX_STEPS="$MAX_STEPS" VAL_CHECK_INTERVAL=999 WARMUP_STEPS=0 LOAD_CHECKPOINT=""
    BINDCMD=""
)
smoke_env_str() { local s=; local kv; for kv in "${smoke_env[@]}"; do s+=" $kv"; done; echo "$s"; }

# edge-case #16: bare-metal pip install only inside venv/conda
bare_prereq_check() {
    say "Bare-metal prerequisite check"
    require_tool python
    require_tool torchrun
    info "python: $(python --version 2>&1)"
    python -c "import torch; print('torch=%s cuda=%s' % (torch.__version__, torch.cuda.is_available()))" \
        || die "torch import failed"
    local miss=()
    for m in nemo megatron transformer_engine hydra mlperf_common pytorch_lightning apex; do
        python -c "import $m" >/dev/null 2>&1 || miss+=("$m")
    done
    if (( ${#miss[@]} > 0 )); then
        err "Missing Python modules: ${miss[*]}"
        info "Minimal: pip install -r $NEMO_DIR/requirements.txt"
        info "Full: NeMo, Megatron-LM, TransformerEngine, apex — build from source (see Dockerfile)."
        if yesno "Run 'pip install -r requirements.txt'?" n; then
            if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
                err "No venv/conda detected. Refusing to pip install into system site-packages."
                info "Create one: python -m venv ~/.venv/mlperf && source ~/.venv/mlperf/bin/activate"
                die "Activate an env and retry."
            fi
            pip install -r "$NEMO_DIR/requirements.txt" || warn "pip install had issues"
        fi
        yesno "Continue despite missing modules?" n || die "Aborted."
    fi
}

# edge-case #13: only called by local bare; srun_bare relies on srun-injected env
bare_env_export() {
    export DATADIR LOGDIR SEED
    export PREPROC_DATA_DIR="$DATADIR/8b"
    : "${RANK:=0}"; : "${LOCAL_RANK:=0}"
    : "${WORLD_SIZE:=1}"; : "${LOCAL_WORLD_SIZE:=1}"
    : "${MASTER_ADDR:=127.0.0.1}"; : "${MASTER_PORT:=$DEFAULT_MPORT}"
    : "${SLURM_JOB_ID:=bare}"; : "${SLURM_PROCID:=$RANK}"; : "${SLURM_LOCALID:=$LOCAL_RANK}"
    : "${BINDCMD:=}"
    export RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
           SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID BINDCMD
}

track_container() { CLEANUP_CONTAINERS+=("$1"); }

emit_prepare_summary() {
    cat <<EOF

========= PREPARE-ONLY SUMMARY =========
repo         : $REPO_DIR
nemo dir     : $NEMO_DIR
image        : ${IMAGE:-<none>}
sqsh         : ${SQSH:-<none>}
cont ref     : ${CONT_REF:-<none>}
datadir      : $DATADIR/8b
logdir       : $LOGDIR
config       : ${CFG_FILE:-<custom smoke>}
GPUs         : ${NGPU:-?} / ${GPU_TOTAL:-?}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})
nnodes       : ${DGXNNODES:-1}   gpus/node : ${DGXNGPU:-$NGPU}
seed         : $SEED
walltime     : ${WALLTIME:-?}
=========================================

To launch later:

# Slurm (containerized, MLPerf-native):
CONT="$CONT_REF" DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \\
  sbatch -N ${DGXNNODES:-1} --time=${WALLTIME:-00:30:00} "$NEMO_DIR/run.sub"

# Single-node docker:
docker run --rm --gpus all --ipc=host --shm-size=16g \\
  -v "$DATADIR/8b:/preproc_data:ro" \\
  -v "$DATADIR/8b/tokenizer:/workspace/llm/nemo_tokenizer:ro" \\
  -v "$LOGDIR:/results" \\
  ${IMAGE:-<image>} bash -c \\
    "cd /workspace/llm && source config_common.sh && source config_common_cg.sh && \\
     source config_common_8b.sh && source ${CFG_FILE:-<cfg>} && bash run_and_time.sh"

# Bare-metal single-node:
cd "$NEMO_DIR" && \\
  source config_common.sh && source config_common_cg.sh && source config_common_8b.sh && \\
  source "${CFG_FILE:-<cfg>}" && \\
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \\
  MASTER_ADDR=127.0.0.1 MASTER_PORT=$DEFAULT_MPORT SEED=$SEED WALLTIME=${WALLTIME:-30} \\
  SLURM_JOB_ID=local SLURM_PROCID=0 SLURM_LOCALID=0 BINDCMD='' \\
  bash run_and_time.sh
EOF
}

# ====================================================================
# dispatch
# ====================================================================
case "$METHOD" in
    prepare)
        emit_prepare_summary
        exit 0
        ;;
    sbatch)
        ACCOUNT="$(ask 'Slurm --account (blank to skip)' '')"
        PARTITION="$(ask 'Slurm --partition (blank to skip)' '')"
        RESERVATION="$(ask 'Slurm --reservation (blank to skip)' '')"
        NEXP="$(ask 'NEXP (repetitions per job)' 1)"
        ARGS=(-N "$DGXNNODES" --time="$WALLTIME")
        [[ -n "$ACCOUNT" ]]     && ARGS+=(--account="$ACCOUNT")
        [[ -n "$PARTITION" ]]   && ARGS+=(--partition="$PARTITION")
        [[ -n "$RESERVATION" ]] && ARGS+=(--reservation="$RESERVATION")
        say "sbatch ${ARGS[*]} run.sub  CONT=$CONT_REF"
        CONT="$CONT_REF" DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" NEXP="$NEXP" \
            sbatch "${ARGS[@]}" run.sub
        ;;
    srun)
        NODES="$(ask 'Nodes (-N)' "$DGXNNODES")"
        NTPN="$(ask 'ntasks-per-node (= GPUs/node)' "$DGXNGPU")"
        GPUS="$(ask 'gpus-per-task' 1)"
        TIME="$(ask 'time limit (HH:MM:SS)' "$WALLTIME")"
        say "srun + pyxis/enroot. CONT=$CONT_REF"
        CONT="$CONT_REF" DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \
            srun -N "$NODES" --ntasks-per-node="$NTPN" --gpus-per-task="$GPUS" --time="$TIME" \
                 --container-image="$CONT_REF" \
                 --container-mounts="$DATADIR/8b:/preproc_data:ro,$DATADIR/8b/tokenizer:/workspace/llm/nemo_tokenizer:ro,$LOGDIR:/results" \
                 --container-workdir=/workspace/llm \
                 bash run_and_time.sh
        ;;
    sbatch_bare)
        bare_prereq_check
        ACCOUNT="$(ask 'Slurm --account (blank to skip)' '')"
        PARTITION="$(ask 'Slurm --partition (blank to skip)' '')"
        WRAP="cd '$NEMO_DIR' && source config_common.sh && source config_common_cg.sh && source config_common_8b.sh && source '$CFG_FILE' && bash run_and_time.sh"
        ARGS=(-N "$DGXNNODES" --ntasks-per-node="$DGXNGPU" --gpus-per-task=1 --time="$WALLTIME")
        [[ -n "$ACCOUNT" ]]   && ARGS+=(--account="$ACCOUNT")
        [[ -n "$PARTITION" ]] && ARGS+=(--partition="$PARTITION")
        DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \
            sbatch "${ARGS[@]}" --wrap="$WRAP"
        ;;
    srun_bare)
        bare_prereq_check
        NODES="$(ask 'Nodes (-N)' "$DGXNNODES")"
        NTPN="$(ask 'ntasks-per-node' "$DGXNGPU")"
        GPUS="$(ask 'gpus-per-task' 1)"
        TIME="$(ask 'time limit (HH:MM:SS)' "$WALLTIME")"
        DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \
            srun -N "$NODES" --ntasks-per-node="$NTPN" --gpus-per-task="$GPUS" --time="$TIME" \
                 --chdir="$NEMO_DIR" \
                 bash -c "source config_common.sh && source config_common_cg.sh && source config_common_8b.sh && source '$CFG_FILE' && bash run_and_time.sh"
        ;;
    docker)
        CNAME="mlperf-llama31-$$-$(date +%s)"
        track_container "$CNAME"
        docker run --name "$CNAME" "${docker_common_args[@]}" \
            "$IMAGE" bash -c "
                cd /workspace/llm &&
                source config_common.sh && source config_common_cg.sh && source config_common_8b.sh &&
                source /workspace/llm/$CFG_FILE &&
                export DGXNGPU=$NGPU DGXNNODES=1 BINDCMD='' &&
                bash run_and_time.sh
            "
        ;;
    bare)
        bare_prereq_check
        bare_env_export
        ln -sfn "$DATADIR/8b/tokenizer" "$NEMO_DIR/nemo_tokenizer"
        (
            cd "$NEMO_DIR"
            set +u; source config_common.sh && source config_common_cg.sh && source config_common_8b.sh; set -u
            set +u; source "$CFG_FILE"; set -u
            export DGXNGPU="$NGPU" DGXNNODES=1 BINDCMD=""
            bash run_and_time.sh
        )
        ;;
    smoke)
        CNAME="mlperf-smoke-$$-$(date +%s)"
        track_container "$CNAME"
        docker run --name "$CNAME" "${docker_common_args[@]}" \
            -e SLURM_JOB_ID=smoke \
            "$IMAGE" bash -c "
                cd /workspace/llm &&
                source config_common.sh && source config_common_cg.sh && source config_common_8b.sh &&
                export $(smoke_env_str) &&
                bash run_and_time.sh
            "
        ;;
    smoke_bare)
        bare_prereq_check
        bare_env_export
        ln -sfn "$DATADIR/8b/tokenizer" "$NEMO_DIR/nemo_tokenizer"
        (
            cd "$NEMO_DIR"
            set +u; source config_common.sh && source config_common_cg.sh && source config_common_8b.sh; set -u
            export "${smoke_env[@]}"
            bash run_and_time.sh
        )
        ;;
    bare_multi)
        bare_prereq_check
        NNODES="$(ask 'Total NNODES' "$DGXNNODES")"
        NODE_RANK="$(ask_req 'This NODE_RANK (0..NNODES-1)')"
        GPUS_PER_NODE="$(ask 'GPUs per node' "$NGPU")"
        M_ADDR="$(ask_req 'MASTER_ADDR (reachable from all nodes)')"
        M_PORT="$(ask 'MASTER_PORT' "$DEFAULT_MPORT")"
        RDZV_ID="$(ask 'rendezvous id (MUST be same on all nodes)' "mlperf-llama31-8b")"
        ln -sfn "$DATADIR/8b/tokenizer" "$NEMO_DIR/nemo_tokenizer"
        export DATADIR LOGDIR SEED PREPROC_DATA_DIR="$DATADIR/8b"
        (
            cd "$NEMO_DIR"
            set +u; source config_common.sh && source config_common_cg.sh && source config_common_8b.sh; set -u
            set +u; source "$CFG_FILE"; set -u
            export DGXNGPU="$GPUS_PER_NODE" DGXNNODES="$NNODES" BINDCMD=""
            say "torchrun multi-node: nnodes=$NNODES node_rank=$NODE_RANK nproc=$GPUS_PER_NODE rdzv=$M_ADDR:$M_PORT"
            torchrun \
                --nnodes="$NNODES" --node_rank="$NODE_RANK" \
                --nproc_per_node="$GPUS_PER_NODE" \
                --rdzv_id="$RDZV_ID" --rdzv_backend=c10d \
                --rdzv_endpoint="$M_ADDR:$M_PORT" \
                pretrain.py
        )
        ;;
    *) die "Unknown method: $METHOD" ;;
esac

ec=$?
if (( ec == 0 )); then
    say "Done. Outputs in: $LOGDIR"
else
    err "Exited with code $ec. Check $LOGDIR for details."
fi
exit "$ec"
