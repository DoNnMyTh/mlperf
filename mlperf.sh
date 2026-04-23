#!/usr/bin/env bash
# Interactive runner for the NVIDIA MLPerf Training v5.1 submissions.
# Workload-agnostic: reads a manifest from workloads/<name>.manifest.sh, then
# runs a six-phase flow (repo → container → dataset → config → runtime → launch).
#
# Usage: bash mlperf.sh
#
# Supported launchers: sbatch run.sub / srun+Pyxis / docker / torchrun / bare-metal / prepare-only.

set -u
set -o pipefail


# --- mlperf.sh common-lib hook -----------------------------------------
_MLPERF_LIB_SOURCED=0
if _LIBDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/lib" && pwd -P)" && [[ -f "$_LIBDIR/common.sh" ]]; then
    # shellcheck source=../lib/common.sh
    source "$_LIBDIR/common.sh"
    _MLPERF_LIB_SOURCED=1
    # Optional modules — missing file is a no-op so branches without them
    # still work. Ordering matters: recipes.sh uses calibrate.sh.
    [[ -f "$_LIBDIR/calibrate.sh" ]] && source "$_LIBDIR/calibrate.sh"
    [[ -f "$_LIBDIR/recipes.sh"   ]] && source "$_LIBDIR/recipes.sh"
    [[ -f "$_LIBDIR/monitor.sh"   ]] && source "$_LIBDIR/monitor.sh"
fi
# Auto-yes / config-file via env only — no flag parsing here to avoid
# clobbering per-tool argv handling.
: "${MLPERF_AUTO_YES:=0}"
if [[ -n "${MLPERF_CONFIG_FILE:-}" && -f "${MLPERF_CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${MLPERF_CONFIG_FILE}"
    MLPERF_AUTO_YES=1
fi
# CLI flag parsing (--dry-run, --resume, --calibrate, --cal-list).
MLPERF_DRY_RUN=0
MLPERF_RESUME=0
MLPERF_CALIBRATE=0
MLPERF_CAL_LIST=0
MLPERF_NO_MONITOR=0
for _a in "$@"; do
    case "$_a" in
        --dry-run)    MLPERF_DRY_RUN=1 ;;
        --resume)     MLPERF_RESUME=1 ;;
        --calibrate)  MLPERF_CALIBRATE=1 ;;
        --cal-list)   MLPERF_CAL_LIST=1 ;;
        --no-monitor) MLPERF_NO_MONITOR=1 ;;
        --help|-h)
            cat <<'EOF'
Usage: bash mlperf.sh [flags]
  --dry-run     Show plan, don't launch.
  --resume      Reuse last session answers.
  --calibrate   Run hardware probe (20-step smoke per TP/CP/FP8 combo)
                and populate ~/.cache/mlperf/calibration.tsv.
  --cal-list    Print calibration cache and exit.
  --no-monitor  Disable live divergence/step-time guard during run.
EOF
            exit 0 ;;
    esac
done
if (( MLPERF_CAL_LIST == 1 )); then
    type cal_list >/dev/null 2>&1 || { echo "calibration lib not loaded"; exit 1; }
    cal_list
    exit 0
fi
(( MLPERF_NO_MONITOR == 1 )) && export MLPERF_MONITOR_ENABLE=0
# Load previous answers so defaults come from the last successful session.
# Only a whitelist of keys is read; ephemeral state (CFG_FILE, METHOD,
# CONT_REF, IMAGE) is reset each run so the wizard doesn't silently reuse
# a stale pick.
_MLPERF_STATE_FILE="${MLPERF_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/mlperf}/session.env"
if [[ -f "$_MLPERF_STATE_FILE" ]]; then
    while IFS= read -r _line; do
        case "$_line" in
            'declare -- REPO_DIR='*|'declare -- DATADIR='*|'declare -- LOGDIR_PREV='*|\
            'declare -- SEED='*|'declare -- WL_NAME='*)
                eval "$_line" 2>/dev/null || true ;;
        esac
    done < "$_MLPERF_STATE_FILE"
fi
# -----------------------------------------------------------------------

# ====================================================================
# bash + TTY guards
# ====================================================================
if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: Bash >= 4 required. Current: $BASH_VERSION" >&2
    echo "  macOS: brew install bash; run with /opt/homebrew/bin/bash" >&2
    exit 1
fi
if [[ ! -t 0 ]]; then
    echo "ERROR: non-interactive stdin. Run in a terminal." >&2
    exit 1
fi

# ====================================================================
# constants
# ====================================================================
SCRIPT_VERSION="3.0"
REPO_URL="https://github.com/mlcommons/training_results_v5.1.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$SCRIPT_DIR/workloads"

# populated from manifest
WL_NAME=""; WL_DISPLAY=""; WL_IMPL_SUBDIR=""; WL_HUB_REPO=""; WL_IMAGE_TAG_BASE=""
declare -a WL_IMAGE_TAG_VARIANTS=()
WL_DATASET_SUBDIR=""; WL_DATASET_SIZE_GB=100
declare -a WL_DATASET_MARKER_FILES=(); declare -a WL_DATASET_MARKER_DIRS=()
WL_DOWNLOAD_SCRIPT=""; WL_DOWNLOAD_ENV=""; WL_DOWNLOAD_HOST_ENV=""
WL_PREPROC_HOST_SUBPATH=""; WL_PREPROC_MOUNT=""
WL_TOKENIZER_HOST_SUBPATH=""; WL_TOKENIZER_MOUNT=""
WL_CONFIG_GLOB="config_*.sh"; WL_CONFIG_EXCLUDE_RE='^config_common|^config_mounts'
WL_ENTRY="run_and_time.sh"; WL_PRETRAIN_PY=""; WL_CONTAINER_WORKDIR="/workspace/llm"
WL_SMOKE_SUPPORTED=0; declare -a WL_SMOKE_ENV=(); declare -a WL_SMOKE_PROMPTS=()
WL_DOC_URL=""; WL_DOCKERFILE_PATCH_FROM=""; WL_DOCKERFILE_PATCH_TO=""

# populated during run
IMAGE=""; SQSH=""; CONT_REF=""; CFG_FILE=""; IS_CUSTOM=0
MAX_STEPS=3; LAYERS=2; NGPU=1
DGXNNODES=1; DGXNGPU=1; WALLTIME="30"
NEED_DOCKER=0; NEED_ENROOT=0; NEED_BARE=0
GPU_TOTAL=0; GPU_LIST=""; declare -a GPU_NAMES=()
declare -a CLEANUP_CONTAINERS=()
RUN_ON_LOGIN_NODE=0
declare -A SMOKE_PROMPT_VALUES=()

# ====================================================================
# ui
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
trap 'err "Aborted by signal."; cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

ask()     { local p="$1" d="${2-}" v=""
    (( MLPERF_AUTO_YES == 1 )) && { echo "${d-}"; return; }
            if [[ -n "$d" ]]; then read -r -p "$p [$d]: " v; echo "${v:-$d}"
            else                   read -r -p "$p: "        v; echo "$v"; fi
          }
ask_req() { local p="$1" v=""
    (( MLPERF_AUTO_YES == 1 )) && { err "required value '$p' not supplied in non-interactive mode"; exit 1; }
            while :; do read -r -p "$p: " v; [[ -n "$v" ]] && { echo "$v"; return; }
                         err "value required"; done
          }
yesno()   { local p="$1" d="${2-y}" v=""
    (( MLPERF_AUTO_YES == 1 )) && { [[ "${d-y}" == "y" ]]; return; }
            while :; do read -r -p "$p (y/n) [$d]: " v; v="${v:-$d}"
                case "$v" in [Yy]|[Yy][Ee][Ss]) return 0;;
                             [Nn]|[Nn][Oo])      return 1;;
                             *) err "Answer y or n";; esac
            done
          }
pick()    { local p="$1"; shift
            local i=1; for o in "$@"; do printf "  [%d] %s\n" "$i" "$o" >&2; i=$((i+1)); done
    (( MLPERF_AUTO_YES == 1 )) && { echo 1; return; }
            local v=""
            while :; do read -r -p "$p [1]: " v; v="${v:-1}"
                [[ "$v" =~ ^[0-9]+$ ]] && (( v>=1 && v<=$# )) && { echo "$v"; return; }
                err "Enter 1..$#"
            done
          }

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
# platform
# ====================================================================
OS="$(uname -s 2>/dev/null || echo unknown)"
case "$OS" in
    Linux*)               PLATFORM=linux   ;;
    Darwin*)              PLATFORM=mac     ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM=windows ;;
    *)                    PLATFORM=unknown ;;
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
# install helpers
# ====================================================================
pkg_install() {
    local p="$1"
    [[ -z "$PKG" ]] && { err "No package manager; install $p manually."; return 1; }
    if [[ -n "$SUDO" ]]; then
        yesno "Run '$SUDO $PKG install $p' (needs admin password)?" y || return 1
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
        mac-git)        echo "    xcode-select --install   or   brew install git" ;;
        mac-docker)     echo "    https://docs.docker.com/desktop/install/mac-install/" ;;
        linux-*)        echo "    Use your distribution's package manager." ;;
        *)              echo "    See official docs for '$tool'." ;;
    esac
}

require_tool() {
    local tool="$1"
    if command -v "$tool" >/dev/null 2>&1; then
        info "$tool: $(command -v "$tool")"
        return 0
    fi
    err "$tool not found."
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

docker_pull_with_auth() {
    local img="$1" tmp rc
    # Fast-path: image already local. Avoids the "Status: Image is up to
    # date ..." round-trip observed on every rerun.
    if docker image inspect "$img" >/dev/null 2>&1; then
        info "Image already local: $img (skipping pull)"
        return 0
    fi
    tmp="/tmp/.pull.$$"
    # Run docker pull separately so we can distinguish "pull failed (auth)"
    # from "pull failed (network/404/disk)". pipefail would otherwise hide
    # non-auth failures behind a failed grep.
    if declare -f retry >/dev/null 2>&1; then
        retry docker pull "$img" >"$tmp" 2>&1; rc=$?
    else
        docker pull "$img" >"$tmp" 2>&1; rc=$?
    fi
    cat "$tmp"
    if (( rc == 0 )); then
        rm -f "$tmp"
        return 0
    fi
    if grep -qE "unauthorized|denied|authentication required|401" "$tmp"; then
        rm -f "$tmp"
        warn "Pull failed — private registry?"
        local host; host="$(awk -F'/' '{print $1}' <<<"$img")"
        [[ "$host" != *.* ]] && host="docker.io"
        yesno "Run 'docker login $host' now?" y || die "Cannot pull without auth."
        docker login "$host" || die "docker login failed"
        docker pull "$img" || die "pull still failed"
    else
        rm -f "$tmp"
        die "docker pull failed (exit $rc) — not an auth error. Check network / image name / disk."
    fi
}

enroot_import_with_auth() {
    local out="$1" ref="$2"
    local dir; dir="$(dirname "$out")"
    [[ -w "$dir" ]] || die "No write permission on $dir."
    if ! enroot import -o "$out" "$ref" 2>/tmp/.enroot.$$; then
        if grep -qE "401|unauthorized|credential" /tmp/.enroot.$$ 2>/dev/null; then
            rm -f /tmp/.enroot.$$
            warn "enroot import auth error."
            info "Configure creds at ~/.config/enroot/.credentials"
            info "Format: machine auth.docker.io login <user> password <token>"
            die "Set credentials and retry."
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

gpu_arch_code() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        [[ -n "$cc" ]] && { echo "$cc"; return; }
    fi
    echo 0
}

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

choose_gpus() {
    detect_gpus
    if (( GPU_TOTAL == 0 )); then
        warn "No GPU visible (login node?)."
        if yesno "Continue (will be set on allocated nodes)?" y; then
            RUN_ON_LOGIN_NODE=1
            NGPU="$(ask 'GPUs per node (target)' "${DGXNGPU:-8}")"
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

free_gb() {
    local path="$1" gb
    if [[ "$PLATFORM" == "mac" ]]; then
        gb=$(df -g "$path" 2>/dev/null | awk 'NR==2 {print $4}')
    else
        gb=$(df -BG "$path" 2>/dev/null | awk 'NR==2 {gsub(/G$/,"",$4); print $4}')
    fi
    gb="${gb//[^0-9]/}"
    echo "${gb:-0}"
}

need_space_gb() {
    local path="$1" need="$2" label="$3"
    local have; have=$(free_gb "$path")
    info "$label: ${have}G free (need ~${need}G)"
    if (( have < need )); then
        warn "Low disk at $path (${have}G < ${need}G)."
        yesno "Continue anyway?" n || die "Free space and retry."
    fi
}

# Generic: find and un-nest $root/$sub/$sub  -> $root/$sub/
fix_nested_dataset() {
    local root="$1" sub="$2" marker="$3"
    if [[ -d "$root/$sub/$sub" && -f "$root/$sub/$sub/$marker" && ! -f "$root/$sub/$marker" ]]; then
        warn "Detected nested $root/$sub/$sub/ — un-nesting."
        if yesno "Move $root/$sub/$sub/* -> $root/$sub/ ?" y; then
            (
                cd "$root/$sub" || exit 1
                shopt -s dotglob nullglob
                local f
                for f in "$sub"/*; do
                    local n; n="$(basename "$f")"
                    [[ -e "$n" && "$n" != "$sub" ]] && rm -rf "$n"
                    mv -- "$f" .
                done
                rmdir "$sub"
            ) || die "Un-nest failed."
            info "Un-nested."
        fi
    fi
}

verify_dataset_md5() {
    local dir="$1"
    command -v md5sum >/dev/null 2>&1 || { warn "md5sum missing; skipping verify."; return; }
    local f total_gb
    # Give the user a runtime estimate — md5 on a 100 GB file at ~500 MB/s
    # is ~3.5 min per file, and there are usually several. A silent hang
    # without this prompt is what makes users ^C.
    total_gb=$(du -sBG "$dir" 2>/dev/null | awk '{gsub(/G/,"",$1); print $1+0}')
    if (( total_gb > 10 )); then
        warn "Dataset at $dir is ${total_gb} GB — md5 will take ~$(( total_gb / 30 + 1 )) min on a fast disk."
        yesno "Proceed with md5 verify?" n || { info "Skipped md5 verify."; return 0; }
    fi
    shopt -s nullglob
    for f in "$dir"/*.md5; do
        info "Checking $(basename "$f")... (Ctrl-C to abort safely)"
        # -q prints only mismatches. Stream the actual md5sum so the user
        # sees per-file progress rather than a single silent prompt.
        ( cd "$dir" && md5sum --check "$(basename "$f")" ) \
            || { err "MD5 mismatch in $f"; shopt -u nullglob; return 1; }
        info "  OK"
    done
    shopt -u nullglob
}

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
track_container() { CLEANUP_CONTAINERS+=("$1"); }

# ====================================================================
# banner
# ====================================================================
cat <<BANNER

╔══════════════════════════════════════════════════════════════╗
║ MLPerf Training v5.1 — interactive runner   v$SCRIPT_VERSION           ║
║ platform: $PLATFORM   pkg: ${PKG:-none}   bash: ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}
╚══════════════════════════════════════════════════════════════╝
BANNER

# ====================================================================
# Auto-detect: probe the system up-front so every prompt below can offer
# a smart default. Honours explicit env overrides.
# ====================================================================
autodetect_defaults() {
    say "Step -1: auto-detecting sensible defaults"

    # -------- REPO_DIR --------
    if [[ -z "${AUTO_REPO_DIR:-}" ]]; then
        for _cand in "${REPO_DIR:-}" "$PWD/training_results_v5.1" \
                     "$HOME/training_results_v5.1" \
                     "/scratch/training_results_v5.1" \
                     "/e/training_results_v5.1-main" \
                     "/data/training_results_v5.1"; do
            [[ -n "$_cand" && -d "$_cand/NVIDIA/benchmarks" ]] && { AUTO_REPO_DIR="$_cand"; break; }
        done
        : "${AUTO_REPO_DIR:=$PWD/training_results_v5.1}"
        info "repo      : $AUTO_REPO_DIR $([[ -d "$AUTO_REPO_DIR/.git" || -d "$AUTO_REPO_DIR/NVIDIA" ]] && echo '(existing)' || echo '(will clone)')"
    fi

    # -------- DATADIR --------
    if [[ -z "${AUTO_DATADIR:-}" ]]; then
        for _cand in "${DATADIR:-}" "/scratch/mlperf_data" "/mnt/mlperf_data" \
                     "/data/mlperf_data" "$PWD/mlperf_data" "/e/mlperf_data"; do
            [[ -n "$_cand" && -d "$_cand" ]] && { AUTO_DATADIR="$_cand"; break; }
        done
        : "${AUTO_DATADIR:=$PWD/mlperf_data}"
        info "datadir   : $AUTO_DATADIR"
    fi

    # -------- LOGDIR --------
    : "${AUTO_LOGDIR:=${LOGDIR:-$PWD/results}}"
    info "logdir    : $AUTO_LOGDIR"

    # -------- GPU arch + image variant --------
    AUTO_GPU_ARCH="$(gpu_arch_code)"
    case "$AUTO_GPU_ARCH" in
        89)            AUTO_IMG_VARIANT="sm89"      ;;
        90)            AUTO_IMG_VARIANT="sm90"      ;;
        100|101|103)   AUTO_IMG_VARIANT="blackwell" ;;
        *)             AUTO_IMG_VARIANT=""          ;;
    esac
    info "gpu arch  : sm_${AUTO_GPU_ARCH:-?}  suggested variant: ${AUTO_IMG_VARIANT:-build-locally}"

    # -------- GPU count + memory (for dynamic config sizing) --------
    AUTO_NGPU=0; AUTO_GPU_MEM_MIB=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        AUTO_NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
        AUTO_GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [[ -z "$AUTO_GPU_MEM_MIB" ]] && AUTO_GPU_MEM_MIB=0
    fi
    info "gpus      : ${AUTO_NGPU}x @ ${AUTO_GPU_MEM_MIB} MiB"

    # -------- Existing locally-cached image --------
    # Runs before workload is selected, so WL_IMAGE_TAG_BASE is empty here
    # on the first pass. Just list any mlperf-nvidia tag; variant-preferred
    # match re-runs later from Step 2 where WL_IMAGE_TAG_BASE is known.
    # Use `--` before patterns starting with '-' so GNU grep doesn't parse
    # them as flags (caught "-sm90$" mis-read as `-s -m 90$`).
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        local _all; _all="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
            | grep -E -- 'mlperf-nvidia' || true)"
        AUTO_LOCAL_IMG=""
        if [[ -n "$AUTO_IMG_VARIANT" && -n "$WL_IMAGE_TAG_BASE" ]]; then
            AUTO_LOCAL_IMG="$(echo "$_all" | grep -E -- "${WL_IMAGE_TAG_BASE}-${AUTO_IMG_VARIANT}\$" | head -n 1)"
        fi
        [[ -z "$AUTO_LOCAL_IMG" ]] && AUTO_LOCAL_IMG="$(echo "$_all" | head -n 1)"
        [[ -n "$AUTO_LOCAL_IMG" ]] && info "local image present: $AUTO_LOCAL_IMG"
    fi

    # -------- Slurm account / partition --------
    : "${AUTO_SLURM_ACCOUNT:=${SLURM_ACCOUNT:-${SBATCH_ACCOUNT:-}}}"
    : "${AUTO_SLURM_PARTITION:=${SLURM_PARTITION:-${SBATCH_PARTITION:-}}}"
    if [[ -z "$AUTO_SLURM_ACCOUNT" ]] && command -v sacctmgr >/dev/null 2>&1; then
        # `timeout` guard — sacctmgr can hang when the slurmdbd is unreachable.
        if command -v timeout >/dev/null 2>&1; then
            AUTO_SLURM_ACCOUNT="$(timeout 5 sacctmgr -n show user "${USER:-}" format=defaultaccount 2>/dev/null | awk '{print $1}' | head -1)"
        fi
    fi
    [[ -n "$AUTO_SLURM_ACCOUNT" ]]   && info "slurm account   : $AUTO_SLURM_ACCOUNT"
    [[ -n "$AUTO_SLURM_PARTITION" ]] && info "slurm partition : $AUTO_SLURM_PARTITION"

    # -------- Preferred launcher --------
    if command -v sbatch >/dev/null 2>&1; then
        AUTO_METHOD="sbatch"
    elif command -v docker >/dev/null 2>&1; then
        AUTO_METHOD="docker"
    else
        AUTO_METHOD="bare"
    fi
    info "launcher  : $AUTO_METHOD (first-viable)"
}

# ====================================================================
# step 0: preflight + workload selection
# ====================================================================
say "Step 0: preflight"
require_tool git
autodetect_defaults

say "Workload selection"
[[ -d "$WORKLOADS_DIR" ]] || die "workloads/ dir not found at $WORKLOADS_DIR"
mapfile -t MANIFESTS < <(ls "$WORKLOADS_DIR"/*.manifest.sh 2>/dev/null)
(( ${#MANIFESTS[@]} > 0 )) || die "No workload manifests found."

declare -a WL_LABELS=() WL_PATHS=()
for mf in "${MANIFESTS[@]}"; do
    name="$(basename "$mf" .manifest.sh)"
    display="$(grep -E '^WL_DISPLAY=' "$mf" | head -1 | sed -E 's/^WL_DISPLAY="?([^"]*)"?/\1/')"
    WL_LABELS+=("$name  —  ${display:-$name}")
    WL_PATHS+=("$mf")
done
sel=$(pick "Choose workload" "${WL_LABELS[@]}")
# shellcheck disable=SC1090
source "${WL_PATHS[$((sel-1))]}"
say "Selected: $WL_DISPLAY"
[[ -n "$WL_DOC_URL" ]] && info "Docs: $WL_DOC_URL"

# ====================================================================
# step 1: repo
# ====================================================================
say "Step 1: repository"
# If autodetect found a checkout, accept it with a single confirmation
# instead of the double prompt the old flow used.
if [[ -d "$AUTO_REPO_DIR/NVIDIA" ]]; then
    # Autodetected clone is the right answer >99% of the time; no prompt.
    info "Using existing checkout: $AUTO_REPO_DIR"
    REPO_DIR="$AUTO_REPO_DIR"
    validate_path "$REPO_DIR" "repo"
elif yesno "Clone mlcommons/training_results_v5.1 into $AUTO_REPO_DIR?" y; then
    REPO_DIR="$AUTO_REPO_DIR"
    validate_path "$REPO_DIR" "repo"
    if [[ -e "$REPO_DIR" ]]; then
        yesno "$REPO_DIR exists. Reuse?" y || die "Pick a different path."
    else
        require_tool git
        retry git clone --depth 1 https://github.com/mlcommons/training_results_v5.1.git "$REPO_DIR" \
            || die "git clone failed."
    fi
else
    die "Cannot proceed without a repo."
fi
IMPL_DIR="$REPO_DIR/$WL_IMPL_SUBDIR"
[[ -f "$IMPL_DIR/Dockerfile" ]] || die "Dockerfile missing at $IMPL_DIR"
cd "$IMPL_DIR" || die "cd to $IMPL_DIR failed"
info "CWD: $PWD"

# ====================================================================
# step 2: container source
# ====================================================================
say "Step 2: container source"
HAS_ENROOT=0; command -v enroot >/dev/null 2>&1 && HAS_ENROOT=1
BARE_OK=1
[[ "$PLATFORM" == "windows" ]] && { BARE_OK=0; info "Windows: bare-metal disabled."; }

OPTS=()
# Ordering: [1] = best default. Cached image → matching-variant pull →
# build-local → other pulls. That way pressing Enter does the right thing
# on a properly-detected host (no re-pull, no unnecessary build).
if [[ -n "${AUTO_LOCAL_IMG:-}" ]]; then
    OPTS+=("docker: use cached $AUTO_LOCAL_IMG")
fi
if [[ -n "${AUTO_IMG_VARIANT:-}" && -n "$WL_HUB_REPO" ]]; then
    for variant in "${WL_IMAGE_TAG_VARIANTS[@]:-}"; do
        [[ "$variant" == "$AUTO_IMG_VARIANT" ]] \
            && OPTS+=("docker: pull $WL_HUB_REPO:$WL_IMAGE_TAG_BASE-$variant")
    done
fi
OPTS+=("docker: build locally (Dockerfile)")
for variant in "${WL_IMAGE_TAG_VARIANTS[@]:-}"; do
    [[ -n "$WL_HUB_REPO" ]] || continue
    [[ "$variant" == "$AUTO_IMG_VARIANT" ]] && continue  # already added above
    if [[ -z "$variant" ]]; then
        OPTS+=("docker: pull $WL_HUB_REPO:$WL_IMAGE_TAG_BASE")
    else
        OPTS+=("docker: pull $WL_HUB_REPO:$WL_IMAGE_TAG_BASE-$variant")
    fi
done
(( BARE_OK == 1 )) && OPTS+=("none — bare-metal (use host Python)")
if (( HAS_ENROOT == 1 )); then
    OPTS+=("enroot: import from registry to .sqsh (no docker)")
    OPTS+=("enroot: use existing .sqsh file")
fi
# Recommend the variant that matches the detected GPU arch.
_rec=""
case "$AUTO_IMG_VARIANT" in
    sm89)       _rec="pull sm89 variant for Ada (sm_89)" ;;
    sm90)       _rec="pull sm90 variant for Hopper (H100/H200)" ;;
    blackwell)  _rec="pull blackwell variant for sm_100/103" ;;
esac
[[ -n "$_rec" ]] && info "Recommendation: $_rec"
[[ -n "${AUTO_LOCAL_IMG:-}" ]] && info "(Tip: local image already cached — $AUTO_LOCAL_IMG)"
sel=$(pick "Container source" "${OPTS[@]}")
CHOICE="${OPTS[$((sel-1))]}"
info "Selected: $CHOICE"

case "$CHOICE" in
    docker:*)  NEED_DOCKER=1 ;;
    enroot:*)  NEED_ENROOT=1 ;;
    none*)     NEED_BARE=1   ;;
esac

if (( NEED_DOCKER == 1 )); then
    require_tool docker; wait_for_docker; check_nvidia
    _local_arch=$(gpu_arch_code)
    # Per-variant sm_* coverage of the published tags. Update when new
    # variants (e.g. -sm90 for Hopper) are pushed to the registry.
    case "$CHOICE" in
        *"-blackwell"*) _covered="100 103" ;;
        *"-sm89"*)      _covered="89 100 103" ;;
        *"-sm90"*)      _covered="89 90 100 103" ;;
        *)              _covered="" ;;
    esac
    if [[ -n "$_covered" && "$_local_arch" != "0" ]]; then
        _hit=0
        for _a in $_covered; do [[ "$_local_arch" == "$_a" ]] && _hit=1; done
        if (( _hit == 0 )); then
            warn "Image built for sm=[$_covered]; detected GPU sm_$_local_arch."
            info "  For H100/H200 (sm_90): build locally and accept the Dockerfile"
            info "  patch prompt so NVTE_CUDA_ARCHS adds '90'."
            yesno "Continue anyway (kernels will fail at runtime)?" n || die "Aborted."
        fi
    fi
    unset _local_arch _covered _hit _a
fi
(( NEED_ENROOT == 1 )) && { require_tool enroot; check_nvidia; }

case "$CHOICE" in
    "docker: build"*)
        IMAGE="$(ask 'Local image name:tag' "mlperf-nvidia:$WL_IMAGE_TAG_BASE")"
        # Serialize Dockerfile patch + docker build against any concurrent
        # driver run in the same IMPL_DIR — otherwise two runs can stomp each
        # other's sed on NVTE_CUDA_ARCHS.
        (
            exec 9>"$IMPL_DIR/.mlperf.build.lock"
            command -v flock >/dev/null 2>&1 \
                && { flock -w 1800 9 || die "another run holds build lock on $IMPL_DIR"; }
        # Smart default for the patch prompt: if the detected local arch is
        # one the patch is meant for (sm_89 Ada, sm_90 Hopper), default to
        # 'y' so the user does not silently get a Blackwell-only image.
        _patch_default="n"
        case "$AUTO_GPU_ARCH" in 89|90) _patch_default="y" ;; esac
        if [[ -n "$WL_DOCKERFILE_PATCH_FROM" ]] \
           && yesno "Patch Dockerfile for broader sm coverage ($WL_DOCKERFILE_PATCH_FROM -> $WL_DOCKERFILE_PATCH_TO)?" "$_patch_default"; then
            # NVTE arch patches: PATCH_TO is of the form
            #   NVTE_CUDA_ARCHS="a;b;c"
            # Use regex replacement on the NVTE_CUDA_ARCHS="..." line so the
            # patch is idempotent AND works regardless of the current arch
            # list (prior sm89 patch, upstream default, etc.).
            if [[ "$WL_DOCKERFILE_PATCH_TO" =~ ^NVTE_CUDA_ARCHS=\"[^\"]*\"$ ]]; then
                if grep -qE 'NVTE_CUDA_ARCHS="[^"]*"' Dockerfile; then
                    sed -i -E "s|NVTE_CUDA_ARCHS=\"[^\"]*\"|$WL_DOCKERFILE_PATCH_TO|" Dockerfile
                    info "Patched (regex) — now: $(grep -oE 'NVTE_CUDA_ARCHS="[^"]*"' Dockerfile | head -1)"
                else
                    warn "No NVTE_CUDA_ARCHS line in Dockerfile; nothing to patch."
                fi
            # Literal string patch (e.g. the dlrm mpi4py RUN rewrite): needs
            # an exact FROM match. If the Dockerfile already contains the TO
            # value, treat as idempotent.
            elif grep -qF "$WL_DOCKERFILE_PATCH_FROM" Dockerfile; then
                sed -i "s|$WL_DOCKERFILE_PATCH_FROM|$WL_DOCKERFILE_PATCH_TO|" Dockerfile
                info "Patched."
            elif grep -qF "$WL_DOCKERFILE_PATCH_TO" Dockerfile; then
                info "Already patched."
            else
                warn "Pattern not found; skipping."
            fi
        fi
        # Post-decision sanity check: does the Dockerfile's NVTE_CUDA_ARCHS
        # include the local GPU's sm? Warn if not — build would succeed but
        # kernels would fail at runtime.
        if [[ "$AUTO_GPU_ARCH" != "0" ]]; then
            _df_archs=$(grep -oE 'NVTE_CUDA_ARCHS="[^"]*"' Dockerfile 2>/dev/null \
                        | head -1 | sed -E 's/.*="([^"]*)".*/\1/')
            if [[ -n "$_df_archs" ]]; then
                _ok=0
                # Match sm_$arch against any of 89/90/100/103/etc. entries,
                # ignoring trailing 'a' (PTX/SASS arch suffix).
                for _a in ${_df_archs//;/ }; do
                    _a_num="${_a%a}"
                    [[ "$_a_num" == "$AUTO_GPU_ARCH" ]] && _ok=1
                done
                if (( _ok == 0 )); then
                    warn "Dockerfile NVTE_CUDA_ARCHS='$_df_archs' does not include sm_$AUTO_GPU_ARCH"
                    warn "Kernels WILL fail at runtime on this GPU."
                    # Auto-yes would silently proceed to a doomed build. Die
                    # explicitly so the batch/CI path surfaces the problem.
                    if (( MLPERF_AUTO_YES == 1 )); then
                        die "Auto-yes + sm mismatch: refuse to proceed. Re-run interactively and accept the arch patch, or set WL_DOCKERFILE_PATCH_TO in the manifest."
                    fi
                    yesno "Continue with this configuration anyway?" n || die "Aborted. Re-run and accept the patch prompt."
                fi
                unset _df_archs _ok _a _a_num
            fi
        fi
        need_space_gb "$(dirname "$IMPL_DIR")" 80 "build dir"
        yesno "Run 'docker build' now?" y || die "Cannot run without an image."
        docker build -t "$IMAGE" . || die "build failed"
        ) || exit $?
        unset _patch_default
        ;;
    "docker: pull"*)
        IMAGE="$(awk '{print $NF}' <<<"$CHOICE")"
        docker_pull_with_auth "$IMAGE"
        ;;
    "docker: use cached"*)
        IMAGE="$(awk '{print $NF}' <<<"$CHOICE")"
        docker image inspect "$IMAGE" >/dev/null 2>&1 \
            || die "Cached image $IMAGE disappeared. Re-run and pick pull/build."
        info "Using cached image: $IMAGE"
        ;;
    none*)
        : ;;
    "enroot: import"*)
        REG="$(ask 'Registry ref' "docker://$WL_HUB_REPO:$WL_IMAGE_TAG_BASE-${WL_IMAGE_TAG_VARIANTS[0]:-}")"
        SQSH_OUT="$(ask 'Output .sqsh path' "$PWD/${WL_NAME}_${WL_IMAGE_TAG_BASE}.sqsh")"
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
esac

if [[ -n "$IMAGE" ]]; then
    # CONT_REF only matters for sbatch/srun launchers. Skip the question
    # entirely when neither is available. Also skip when IMAGE is a local
    # tag (no '/'): pyxis needs a registry reference, and the docker:// /
    # enroot-style '+' variants both collapse to the same string.
    if (command -v sbatch >/dev/null 2>&1 || command -v srun >/dev/null 2>&1) \
       && [[ "$IMAGE" == */* ]]; then
        fmt=$(pick "Pyxis ref format" "docker://$IMAGE" "${IMAGE//\//+}" "skip (not using pyxis)")
        case "$fmt" in
            1) CONT_REF="docker://$IMAGE" ;;
            2) CONT_REF="${IMAGE//\//+}" ;;
            3) CONT_REF="" ;;
        esac
    else
        CONT_REF=""
        [[ "$IMAGE" != */* ]] && info "Local image (no registry) — pyxis ref not applicable."
    fi
elif [[ -n "$SQSH" ]]; then
    CONT_REF="$SQSH"
fi
[[ -n "$CONT_REF" ]] && info "CONT: $CONT_REF"

# ====================================================================
# step 3: dataset
# ====================================================================
say "Step 3: dataset ($WL_DATASET_SUBDIR, ~${WL_DATASET_SIZE_GB}G)"
DATADIR="$(ask 'DATADIR host path' "${DATADIR:-$AUTO_DATADIR}")"
validate_path "$DATADIR" "DATADIR"
# Recursive normalize: strip any chain of trailing /$WL_DATASET_SUBDIR
# segments as long as the parent still has the marker file. Handles
# .../mlperf_data/8b, .../mlperf_data/8b/8b, .../8b/8b/8b, etc.
DATADIR="${DATADIR%/}"
_marker="${WL_DATASET_MARKER_FILES[0]:-}"
while [[ "$DATADIR" == */"$WL_DATASET_SUBDIR" ]] \
      && [[ -n "$_marker" && -f "${DATADIR%/"$WL_DATASET_SUBDIR"}/$WL_DATASET_SUBDIR/$_marker" ]]; do
    warn "DATADIR ends in /$WL_DATASET_SUBDIR — stripping duplicate segment."
    DATADIR="${DATADIR%/"$WL_DATASET_SUBDIR"}"
done
unset _marker
info "DATADIR (normalized): $DATADIR"
# No prompt if it already exists. Create silently otherwise — keeps the
# auto-flow moving without a yes/no on a trivial mkdir.
[[ -d "$DATADIR" ]] || { mkdir -p "$DATADIR" || die "Cannot create $DATADIR"; info "Created $DATADIR"; }
export DATADIR

# un-nest if the first marker file indicates duplicate-cleanup nesting
if (( ${#WL_DATASET_MARKER_FILES[@]} > 0 )); then
    fix_nested_dataset "$DATADIR" "$WL_DATASET_SUBDIR" "${WL_DATASET_MARKER_FILES[0]}"
fi

dataset_files_ok() {
    local f d
    for f in "${WL_DATASET_MARKER_FILES[@]:-}"; do
        [[ -f "$DATADIR/$WL_DATASET_SUBDIR/$f" ]] || return 1
    done
    for d in "${WL_DATASET_MARKER_DIRS[@]:-}"; do
        [[ -d "$DATADIR/$WL_DATASET_SUBDIR/$d" ]] || return 1
    done
    return 0
}

DO_DL=1
if dataset_files_ok; then
    info "Dataset markers present at $DATADIR/$WL_DATASET_SUBDIR — skipping download."
    DO_DL=0
    yesno "Verify md5 checksums? (slow, runs md5sum over the tree)" n \
        && verify_dataset_md5 "$DATADIR/$WL_DATASET_SUBDIR"
fi

if (( DO_DL == 1 )); then
    need_space_gb "$DATADIR" "$WL_DATASET_SIZE_GB" "DATADIR"
    if [[ -z "$WL_DOWNLOAD_SCRIPT" ]]; then
        warn "This workload has no automated download script."
        info "Please stage data manually at: $DATADIR/$WL_DATASET_SUBDIR/"
        yesno "Continue with possibly missing dataset?" n || die "Aborted."
    elif yesno "Run $WL_DOWNLOAD_SCRIPT now?" n; then
      # Lock DATADIR so two concurrent runs can't double-download or race
      # on the same partial dataset tree.
      (
        exec 9>"$DATADIR/.mlperf.download.lock"
        command -v flock >/dev/null 2>&1 \
            && { flock -w 3600 9 || die "another run holds download lock on $DATADIR"; }
        # Expand single layer of ${VAR} references in the manifest's env
        # strings without invoking `eval` on arbitrary content.
        _expand_env() {
            local s="$1"
            # Only expand bare ${NAME} patterns of our exported vars.
            s="${s//\$DATADIR/$DATADIR}"
            s="${s//\${DATADIR\}/$DATADIR}"
            echo "$s"
        }
        if   [[ -n "$IMAGE" ]]; then
            env_line="$(_expand_env "$WL_DOWNLOAD_ENV")"
            docker run --rm --network=host \
                -v "$DATADIR:/data" \
                -e "$env_line" \
                "$IMAGE" bash "$WL_DOWNLOAD_SCRIPT" || die "download failed"
        elif [[ -n "$SQSH" ]]; then
            env_line="$(_expand_env "$WL_DOWNLOAD_ENV")"
            enroot start --mount "$DATADIR:/data" --env "$env_line" \
                "$SQSH" bash "$WL_CONTAINER_WORKDIR/$WL_DOWNLOAD_SCRIPT" || die "download failed"
        else
            require_tool curl; require_tool wget
            host_env="$(_expand_env "$WL_DOWNLOAD_HOST_ENV")"
            (
                cd "$IMPL_DIR"
                # Split "KEY=VALUE" on the first '='; export without eval.
                key="${host_env%%=*}"
                val="${host_env#*=}"
                export "$key=$val"
                bash "$WL_DOWNLOAD_SCRIPT"
            ) || die "download failed"
        fi
        (( ${#WL_DATASET_MARKER_FILES[@]} > 0 )) && \
            fix_nested_dataset "$DATADIR" "$WL_DATASET_SUBDIR" "${WL_DATASET_MARKER_FILES[0]}"
      ) || exit $?
    else
        warn "Skipped download."
        yesno "Continue without dataset?" n || die "Aborted."
    fi
fi

# ====================================================================
# step 4: topology + config (dynamic)
# ====================================================================
say "Step 4: topology"

# Dynamic config emitter. Shape: NNODES × NGPU. Parallelism = largest
# power-of-2 dividing world size, TP-first then CP. Not MLCommons-compliant
# — uses MINIBS=1, reduced WARMUP — but shape-matches upstream config_*.sh
# so run.sub / bash entry both work.
emit_auto_config() {
    local nnodes="$1" ngpu_per_node="$2" gpu_mem_mib="$3" gpu_arch="$4" out="$5"
    local run_mode="${6:-preview}"   # preview | full
    local world=$(( nnodes * ngpu_per_node ))

    # Parallelism: largest power-of-2 TP dividing world size, cap 8; spill
    # excess into CP. PP kept 1 on single-node (no benefit <= 1 node).
    local tp=1 pp=1 cp=1
    while (( tp * 2 <= world && (world % (tp * 2)) == 0 )); do tp=$((tp * 2)); done
    if (( tp > 8 )); then cp=$(( tp / 8 )); tp=8; fi
    # Multi-node: prefer PP over CP once world >= 16 to reduce activation-
    # memory pressure per stage.
    if (( nnodes >= 2 && world >= 16 )); then
        pp=$(( cp )); cp=1
        (( pp < 1 )) && pp=1
    fi

    # Precision from arch capability.
    local fp8=False fp8_hybrid=False
    case "$gpu_arch" in 89|90|100|103) fp8=True; fp8_hybrid=True ;; esac
    local fp4=False
    case "$gpu_arch" in 100|103) fp4=True ;; esac

    # Micro-batch from GPU memory. Empirical: 8B model at bf16 + FP8 needs
    # ~22 GB per GPU per sample after TP sharding. Scale MBS accordingly,
    # capped at 4 for stability on preview runs.
    local mbs=1
    if (( gpu_mem_mib >= 140000 )); then mbs=4      # H200 141GB
    elif (( gpu_mem_mib >= 80000 )); then mbs=2     # H100 / A100 80GB
    else mbs=1
    fi
    local minibs=$(( mbs * world ))

    # MAX_STEPS: preview caps fast; full defers to upstream compliance steps.
    local max_steps val_check warmup
    if [[ "$run_mode" == "preview" ]]; then
        max_steps=100; val_check=50; warmup=2
    else
        max_steps=""; val_check=""; warmup=""
    fi
    cat > "$out" <<EOF
# AUTO-GENERATED by mlperf.sh on $(date -Iseconds)
# Host: $(hostname)  Topology: ${nnodes} node × ${ngpu_per_node} GPU
# GPU: sm_${gpu_arch}, ${gpu_mem_mib} MiB each
# Parallelism: TP=$tp PP=$pp CP=$cp (world=$world)  NOT MLCommons-compliant.
source \$(dirname \${BASH_SOURCE[0]})/config_common.sh
[[ -f \$(dirname \${BASH_SOURCE[0]})/config_common_cg.sh ]] && \
    source \$(dirname \${BASH_SOURCE[0]})/config_common_cg.sh
[[ -f \$(dirname \${BASH_SOURCE[0]})/config_common_8b.sh ]] && \
    source \$(dirname \${BASH_SOURCE[0]})/config_common_8b.sh

export MINIBS=$minibs
export MICRO_BATCH_SIZE=$mbs
export TENSOR_MODEL_PARALLEL=$tp
export PIPELINE_MODEL_PARALLEL=$pp
export CONTEXT_PARALLEL=$cp
export SEQ_PARALLEL=$( [[ $tp -gt 1 ]] && echo True || echo False )
export INTERLEAVED_PIPELINE=null
export FP8=$fp8
export FP8_HYBRID=$fp8_hybrid
export FP4=$fp4
export TP_COMM_OVERLAP=False

$( [[ -n "$warmup" ]] && echo "export WARMUP_STEPS=$warmup" )
$( [[ -n "$max_steps" ]] && echo "export MAX_STEPS=$max_steps" )
$( [[ -n "$val_check" ]] && echo "export VAL_CHECK_INTERVAL=$val_check" )
$( [[ -n "$max_steps" ]] && echo "# preview mode: MAX_STEPS capped. Compliance needs run_mode=full." )

export DGXNNODES=$nnodes
export DGXNGPU=$ngpu_per_node
export DGXSYSTEM=\$(basename \$(readlink -f \${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh\$//')
export WALLTIME_RUNANDTIME=30
export WALLTIME=\$((5 + \${NEXP:-1} * (\$WALLTIME_RUNANDTIME + 5)))
EOF
}

# Topology: 100% auto-detected from cluster state. No prompts.
#   - nodes: SLURM_JOB_NUM_NODES if in allocation, else 1
#   - gpus per node: nvidia-smi count on this host
#   - run mode: env MLPERF_RUN_MODE (preview|full); default preview
#   - every emit_auto_config parameter derived from those three + arch/mem
CFG_FILE=""; IS_CUSTOM=0; AUTO_CFG=""

# Override points (env) so CI/non-interactive runs can still steer:
#   MLPERF_RUN_MODE=full        compliance run, defer MAX_STEPS to upstream
#   MLPERF_NNODES / MLPERF_NGPU_PER_NODE  explicit override of detection
#   MLPERF_TOPOLOGY=advanced    fall back to the upstream-config picker
#   MLPERF_TOPOLOGY=smoke       1-GPU custom smoke
run_mode="${MLPERF_RUN_MODE:-preview}"
topo="${MLPERF_TOPOLOGY:-auto}"

case "$topo" in
    advanced)
        mapfile -t CONFIGS < <(ls $WL_CONFIG_GLOB 2>/dev/null | grep -Ev "$WL_CONFIG_EXCLUDE_RE" || true)
        (( ${#CONFIGS[@]} > 0 )) || die "No upstream config files matching '$WL_CONFIG_GLOB' in $PWD"
        sel=$(pick "Pick an upstream config" "${CONFIGS[@]}")
        CFG_FILE="${CONFIGS[$((sel-1))]}"
        ;;
    smoke)
        IS_CUSTOM=1
        ;;
    auto|*)
        # Detect nodes. In a Slurm allocation SLURM_JOB_NUM_NODES is set.
        nnodes="${MLPERF_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
        ngpu_per_node="${MLPERF_NGPU_PER_NODE:-$AUTO_NGPU}"
        (( ngpu_per_node > 0 )) || die "No GPUs detected; set MLPERF_NGPU_PER_NODE to override."
        say "Step 4: topology (auto)"
        info "nodes        : $nnodes$( [[ -n "${SLURM_JOB_NUM_NODES:-}" ]] && echo ' (from Slurm allocation)')"
        info "gpus/node    : $ngpu_per_node"
        info "gpu arch     : sm_${AUTO_GPU_ARCH:-?}  mem: ${AUTO_GPU_MEM_MIB:-?} MiB"
        info "run mode     : $run_mode   (MLPERF_RUN_MODE=full for compliance)"
        AUTO_CFG="$IMPL_DIR/config_AUTO_${nnodes}x${ngpu_per_node}_${run_mode}_${WL_NAME}.sh"
        emit_auto_config "$nnodes" "$ngpu_per_node" "${AUTO_GPU_MEM_MIB:-0}" "${AUTO_GPU_ARCH:-0}" "$AUTO_CFG" "$run_mode"
        CFG_FILE="$AUTO_CFG"
        info "Generated: $(basename "$AUTO_CFG")"
        info "Sizing TP/PP/CP and MICRO_BATCH from detected hardware — no hardcoded cluster profiles."
        ;;
esac

# ---------------------------------------------------------------
# Runtime estimator — table + cache, used pre-launch so the user sees
# train/eval/total breakdown and a convergence verdict before committing.
#
# step_time_table[workload|arch] = seconds per training step at GBS=1 TP=world.
# Populated from observed runs; update as you gather more data. Cache file
# records measured values post-run so future estimates become accurate.
# ---------------------------------------------------------------
: "${MLPERF_CACHE_DIR:=${XDG_CACHE_HOME:-$HOME/.cache}/mlperf}"
MLPERF_STEP_CACHE="$MLPERF_CACHE_DIR/step_times.tsv"

declare -A _STEP_TIME_TABLE=(
    [llama31_8b|89]=0.40   [llama31_8b|90]=0.20   [llama31_8b|100]=0.10  [llama31_8b|103]=0.08
    [llama31_405b|90]=4.50 [llama31_405b|100]=2.20 [llama31_405b|103]=1.80
    [llama2_70b_lora|89]=1.20 [llama2_70b_lora|90]=0.60 [llama2_70b_lora|100]=0.30 [llama2_70b_lora|103]=0.25
    [flux1|89]=0.35        [flux1|90]=0.18        [flux1|100]=0.09       [flux1|103]=0.07
    [retinanet|89]=0.25    [retinanet|90]=0.12    [retinanet|100]=0.06   [retinanet|103]=0.05
    [rgat|89]=0.30         [rgat|90]=0.15         [rgat|100]=0.08        [rgat|103]=0.06
    [dlrm_dcnv2|89]=0.20   [dlrm_dcnv2|90]=0.10   [dlrm_dcnv2|100]=0.05  [dlrm_dcnv2|103]=0.04
)
declare -A _EVAL_TIME_TABLE=(
    [llama31_8b]=70        [llama31_405b]=600     [llama2_70b_lora]=40
    [flux1]=30             [retinanet]=45         [rgat]=20             [dlrm_dcnv2]=15
)

_step_time_lookup() {
    local wl="$1" arch="$2" world="$3" key="$wl|$arch|$world" per_step=""
    if [[ -f "$MLPERF_STEP_CACHE" ]]; then
        per_step=$(awk -F'\t' -v k="$key" '$1==k {print $2; exit}' "$MLPERF_STEP_CACHE" 2>/dev/null)
        [[ -n "$per_step" ]] && { printf '%s\tcached\n' "$per_step"; return; }
    fi
    per_step="${_STEP_TIME_TABLE[$wl|$arch]:-}"
    if [[ -n "$per_step" && "$world" -gt 1 ]]; then
        per_step=$(awk -v p="$per_step" -v w="$world" 'BEGIN{printf "%.4f", p / (w^0.85)}')
    fi
    [[ -z "$per_step" ]] && { printf '\tunknown\n'; return; }
    printf '%s\ttable\n' "$per_step"
}

_step_time_record_unlocked() {
    local key="$1" per_step="$2"
    local tmp="$MLPERF_STEP_CACHE.tmp.$$"
    [[ -f "$MLPERF_STEP_CACHE" ]] && awk -F'\t' -v k="$key" '$1!=k' "$MLPERF_STEP_CACHE" > "$tmp" || : > "$tmp"
    printf '%s\t%s\t%s\n' "$key" "$per_step" "$(date -Iseconds)" >> "$tmp"
    mv "$tmp" "$MLPERF_STEP_CACHE"
}

_step_time_record() {
    local wl="$1" arch="$2" world="$3" per_step="$4"
    mkdir -p "$MLPERF_CACHE_DIR"
    local key="$wl|$arch|$world"
    if type with_lock >/dev/null 2>&1; then
        with_lock "$MLPERF_STEP_CACHE.lock" 30 _step_time_record_unlocked "$key" "$per_step"
    else
        _step_time_record_unlocked "$key" "$per_step"
    fi
}

# Parse train_step_time values from MLlog JSON log lines under $1 and record
# the median to the step-time cache for future estimates. Skips first 5% of
# samples (warm-up). No-op if <10 samples found.
_record_step_time_from_log() {
    local logdir="$1" wl="$2" arch="$3" world="$4"
    [[ -d "$logdir" ]] || return
    local vals median
    vals=$(grep -rh '"train_step_time"' "$logdir" 2>/dev/null \
        | sed -nE 's/.*"train_step_time"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p')
    local n; n=$(printf '%s\n' "$vals" | grep -c .) || true
    (( n >= 10 )) || return
    local skip=$(( n / 20 ))   # drop first 5% as warm-up
    median=$(printf '%s\n' "$vals" | tail -n +$((skip + 1)) | sort -g \
        | awk '{a[NR]=$1} END{ if(NR==0) exit; if(NR%2) print a[(NR+1)/2]; else printf "%.6f\n",(a[NR/2]+a[NR/2+1])/2 }')
    [[ -z "$median" ]] && return
    _step_time_record "$wl" "$arch" "$world" "$median"
    info "Cached measured step-time: ${median}s/step → $MLPERF_STEP_CACHE"
}

_fmt_duration() {
    local s="${1:-0}"
    s=${s%.*}
    (( s <= 0 )) && { printf '0m\n'; return; }
    local h=$(( s / 3600 )) m=$(( (s % 3600) / 60 ))
    (( h > 0 )) && printf '%dh%02dm\n' "$h" "$m" || printf '%dm\n' "$m"
}

# Parse MAX_STEPS / VAL_CHECK_INTERVAL from upstream config_common*.sh.
# Returns: "<max_steps> <val_check_interval>" or "" if unresolved.
_parse_upstream_steps() {
    local dir="$1" f ms="" vci=""
    for f in "$dir/config_common_8b.sh" "$dir/config_common.sh" "$dir/config_common_cg.sh"; do
        [[ -f "$f" ]] || continue
        [[ -z "$ms" ]]  && ms=$(grep -E '^(export[[:space:]]+)?MAX_STEPS=' "$f" | tail -1 | sed -E 's/.*MAX_STEPS=([0-9]+).*/\1/')
        [[ -z "$vci" ]] && vci=$(grep -E '^(export[[:space:]]+)?VAL_CHECK_INTERVAL=' "$f" | tail -1 | sed -E 's/.*VAL_CHECK_INTERVAL=([0-9]+).*/\1/')
    done
    printf '%s %s\n' "${ms:-}" "${vci:-}"
}

# Emit multi-line estimate block. Sets globals EST_TOTAL_S, EST_VERDICT for
# downstream pre-launch banner.
_estimate_wall_detailed() {
    local wl="$1" arch="$2" world="$3" max_steps="$4" vci="$5" gbs="$6" is_auto="$7"
    local stp_out per_step src
    stp_out=$(_step_time_lookup "$wl" "$arch" "$world")
    per_step="${stp_out%%$'\t'*}"
    src="${stp_out#*$'\t'}"
    local eval_s="${_EVAL_TIME_TABLE[$wl]:-60}"

    if [[ -z "$per_step" || -z "$max_steps" ]]; then
        EST_TOTAL_S=0
        printf '  step-time  : unknown (no table entry for sm_%s)\n' "$arch"
        printf '  max-steps  : %s   val-interval: %s\n' "${max_steps:-?}" "${vci:-?}"
        printf '  ETA        : unknown — run anyway to populate cache\n'
        EST_VERDICT="unknown"
        return
    fi
    local train_s=$(awk -v s="$max_steps" -v p="$per_step" 'BEGIN{printf "%.0f", s*p}')
    local n_evals=0
    (( ${vci:-0} > 0 )) && n_evals=$(( max_steps / vci ))
    local eval_total=$(( n_evals * eval_s ))
    local total=$(( train_s + eval_total ))
    EST_TOTAL_S="$total"

    printf '  step-time  : %ss/step (%s, sm_%s, world=%d)\n' "$per_step" "$src" "$arch" "$world"
    printf '  max-steps  : %d   val-interval: %s (%d evals × %ds)\n' \
        "$max_steps" "${vci:-?}" "$n_evals" "$eval_s"
    printf '  ETA        : train %s + eval %s = %s\n' \
        "$(_fmt_duration "$train_s")" "$(_fmt_duration "$eval_total")" "$(_fmt_duration "$total")"

    # Convergence verdict
    EST_VERDICT="compliant"
    local verdict_msg=""
    if (( is_auto == 1 )); then
        EST_VERDICT="non-compliant"
        verdict_msg="AUTO config (MINIBS=1, GBS=$gbs) — shape-test only, will NOT reach convergence target."
    elif [[ "$wl" == "llama31_405b" && "$world" -lt 512 ]]; then
        EST_VERDICT="infeasible"
        verdict_msg="405B needs ≥512 GPUs for compliance; $world GPUs will take days."
    fi
    [[ -n "$verdict_msg" ]] && printf '  verdict    : %s\n' "$verdict_msg"
}

# Estimator defined here; invocation deferred until after CFG_FILE is sourced
# (it needs MAX_STEPS/VAL_CHECK_INTERVAL/MINIBS/TP etc. from config env).
EST_TOTAL_S=0; EST_VERDICT="unknown"
run_runtime_estimate() {
    (( IS_CUSTOM == 1 )) && { EST_VERDICT="smoke"; return; }
    [[ -z "${CFG_FILE:-}" ]] && return
    local world="${NGPU:-${AUTO_NGPU:-1}}"
    local arch="${AUTO_GPU_ARCH:-90}"
    local is_auto=0
    [[ "$(basename "$CFG_FILE")" == config_AUTO_* ]] && is_auto=1
    local ms="${MAX_STEPS:-}" vci="${VAL_CHECK_INTERVAL:-}"
    if [[ -z "$ms" || -z "$vci" ]]; then
        read -r _ms2 _vci2 <<< "$(_parse_upstream_steps "$(dirname "$CFG_FILE")")"
        [[ -z "$ms" ]]  && ms="$_ms2"
        [[ -z "$vci" ]] && vci="$_vci2"
    fi
    local tp="${TENSOR_MODEL_PARALLEL:-1}" pp="${PIPELINE_MODEL_PARALLEL:-1}" cp="${CONTEXT_PARALLEL:-1}"
    local gbs=$(( ${MINIBS:-1} * world / (tp * pp * cp) ))
    (( gbs < 1 )) && gbs=1
    say "Estimated runtime"
    _estimate_wall_detailed "$WL_NAME" "$arch" "$world" "$ms" "$vci" "$gbs" "$is_auto"
}

# ====================================================================
# step 5: runtime params
# ====================================================================
say "Step 5: runtime parameters"
LOGDIR_PARENT="$(ask 'LOGDIR' "${LOGDIR_PREV:-$AUTO_LOGDIR}")"
validate_path "$LOGDIR_PARENT" "LOGDIR"
# Auto-create. A yes/no on mkdir is noise when the user just confirmed
# the path by pressing Enter.
[[ -d "$LOGDIR_PARENT" ]] || { mkdir -p "$LOGDIR_PARENT" || die "Cannot create $LOGDIR_PARENT"; info "Created $LOGDIR_PARENT"; }
# Always timestamp so multiple runs never clobber each other. Deferred
# mkdir until right before launch so aborted prereq checks don't litter.
LOGDIR="$LOGDIR_PARENT/$(date +%Y%m%d_%H%M%S)_$$"
info "LOGDIR will be: $LOGDIR (created at launch)"
export LOGDIR LOGDIR_PARENT

while :; do
    SEED="$(ask 'SEED (positive integer)' "${SEED:-42}")"
    [[ "$SEED" =~ ^[0-9]+$ ]] && break
    err "SEED must be a non-negative integer."
done

if (( IS_CUSTOM == 1 )); then
    for prompt in "${WL_SMOKE_PROMPTS[@]}"; do
        key="${prompt%%:*}"; default="${prompt#*:}"
        val="$(ask "$key" "$default")"
        SMOKE_PROMPT_VALUES["$key"]="$val"
    done
    choose_gpus
    # Smoke supports multi-GPU. Split world across TP (powers of 2) so the
    # un-reduced 8B model (if monkey-patch misses) at least shards across
    # devices — 1×H200 cannot hold 8B+optimizer states, 4×H200 can.
    _stp=1
    while (( _stp * 2 <= NGPU && (NGPU % (_stp * 2)) == 0 )); do _stp=$((_stp * 2)); done
    SMOKE_PROMPT_VALUES[TENSOR_MODEL_PARALLEL]="$_stp"
    SMOKE_PROMPT_VALUES[DGXNGPU]="$NGPU"
    info "Smoke will use NGPU=$NGPU (TP=$_stp PP=1 CP=1)"
    unset _stp
else
    set +u
    # shellcheck disable=SC1090
    source "$CFG_FILE"
    set -u
    info "Config: DGXNNODES=${DGXNNODES:-?} DGXNGPU=${DGXNGPU:-?} WALLTIME=${WALLTIME:-?}"
    info "Parallelism: TP=${TENSOR_MODEL_PARALLEL:-1} PP=${PIPELINE_MODEL_PARALLEL:-1} CP=${CONTEXT_PARALLEL:-1}"
    choose_gpus

    # World size = NGPU for local docker/bare (DGXNNODES=1 override). For
    # sbatch/srun the full cluster world size is used; we still validate
    # against NGPU as a local-run sanity check.
    _tp="${TENSOR_MODEL_PARALLEL:-1}"
    _pp="${PIPELINE_MODEL_PARALLEL:-1}"
    _cp="${CONTEXT_PARALLEL:-1}"
    _mp=$(( _tp * _pp * _cp ))
    if (( _mp > NGPU )) && [[ "$RUN_ON_LOGIN_NODE" != "1" ]]; then
        warn "Config needs TP*PP*CP=$_mp GPUs, but you picked $NGPU."
        warn "World size < model parallelism would fail at NCCL init."
        if yesno "Auto-adapt parallelism to fit $NGPU GPUs?" y; then
            # Reduce CP first, then PP, then TP — each must remain a power
            # of 2 (or 1) and their product ≤ NGPU.
            while (( _cp > 1 && _tp * _pp * _cp > NGPU )); do _cp=$(( _cp / 2 )); done
            while (( _pp > 1 && _tp * _pp * _cp > NGPU )); do _pp=$(( _pp / 2 )); done
            while (( _tp > 1 && _tp * _pp * _cp > NGPU )); do _tp=$(( _tp / 2 )); done
            (( _tp * _pp * _cp <= NGPU )) \
                || die "Could not reduce TP*PP*CP to fit $NGPU GPUs; pick a different config or edit manually."
            export TENSOR_MODEL_PARALLEL="$_tp"
            export PIPELINE_MODEL_PARALLEL="$_pp"
            export CONTEXT_PARALLEL="$_cp"
            # Interleaved pipeline only valid when PP>1.
            (( _pp == 1 )) && export INTERLEAVED_PIPELINE=0
            # SEQ_PARALLEL requires TP>1.
            (( _tp == 1 )) && export SEQ_PARALLEL=False
            info "Adapted: TP=$_tp PP=$_pp CP=$_cp (mp=$((_tp*_pp*_cp)), dp=$((NGPU/(_tp*_pp*_cp))))"
        else
            die "World size ($NGPU) < TP*PP*CP ($_mp); will fail at runtime."
        fi
    fi
    unset _tp _pp _cp _mp
fi

# ----------------------------------------------------------------------
# Dynamic calibration + recipe menu (new path).
# Runs when --calibrate was passed OR when the calibration cache already
# has measurements for this (workload, GPU model, NGPU). Replaces the
# static step-time table as the authoritative ETA source.
# ----------------------------------------------------------------------
REC_CHOICE_NAME=""
if type cal_inventory_hw >/dev/null 2>&1; then
    cal_inventory_hw
    if (( MLPERF_CALIBRATE == 1 )); then
        mkdir -p "$LOGDIR_PARENT"
        _cal_log="$LOGDIR_PARENT/calibrate_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$_cal_log"
        info "Running calibration probe → $_cal_log"
        if [[ -n "${IMAGE:-}" && -n "${DATADIR:-}" && -n "${IMPL_DIR:-}" ]]; then
            cal_probe "$WL_NAME" "$NGPU" "$IMAGE" "$DATADIR" "$_cal_log" \
                      "$IMPL_DIR" "docker" || warn "Calibration completed with errors."
        else
            warn "Calibration needs IMAGE+DATADIR+IMPL_DIR — skipping."
        fi
    fi
    if type rec_menu >/dev/null 2>&1 \
       && [[ -n "${CAL_GPU_MODEL:-}" ]] \
       && cal_cache_rows_for "$WL_NAME" "$NGPU" | grep -q .; then
        say "Dynamic recipes available (from calibration cache)"
        if rec_menu "$WL_NAME" "$NGPU"; then
            info "Recipe will override the following values from '$(basename "${CFG_FILE:-none}")':"
            info "  MAX_STEPS, LR, WARMUP_STEPS, VAL_CHECK_INTERVAL,"
            info "  TP/PP/CP, MICRO_BATCH_SIZE, MINIBS, FP8, FP8_HYBRID,"
            info "  SEQ_PARALLEL, GRADIENT_CLIP_VAL."
            info "All other config vars (optimizer, scheduler, dataset) remain from the config file."
            rec_export
            # Flag that a dynamic recipe supplied overrides. Downstream
            # launchers (docker/sbatch/srun) use REC_EXPORT_VARS to forward
            # these via -e or env= prefix. IS_CUSTOM stays off so the user
            # retains access to Slurm launchers for compliant runs.
            REC_OVERRIDES_APPLIED=1
            REC_EXPORT_VARS=(MAX_STEPS VAL_CHECK_INTERVAL WARMUP_STEPS LR
                             TENSOR_MODEL_PARALLEL PIPELINE_MODEL_PARALLEL
                             CONTEXT_PARALLEL MICRO_BATCH_SIZE MINIBS
                             FP8 FP8_HYBRID SEQ_PARALLEL GRADIENT_CLIP_VAL)
            # Mirror to SMOKE_PROMPT_VALUES so the legacy smoke-env codepath
            # (build_smoke_env_str) also picks them up when method=smoke.
            for _k in "${REC_EXPORT_VARS[@]}"; do
                SMOKE_PROMPT_VALUES["$_k"]="${!_k}"
            done
            SMOKE_PROMPT_VALUES[DGXNGPU]="$NGPU"
            EST_TOTAL_S="$REC_CHOICE_ETA_S"
            EST_VERDICT="dynamic"
        fi
    else
        info "(no calibration data yet — using static estimator. Run with --calibrate to populate.)"
    fi
fi

# Now that CFG_FILE (incl. upstream chain-sources) has populated MAX_STEPS
# etc. for non-smoke, compute the runtime estimate for the user.
[[ -z "${EST_TOTAL_S:-}" ]] && run_runtime_estimate

# Only mention WALLTIME units when a Slurm launcher is in play (different
# unit convention between docker/bare and sbatch/srun).
if command -v sbatch >/dev/null 2>&1 || command -v srun >/dev/null 2>&1; then
    info "Note: WALLTIME = MINUTES for docker/bare; HH:MM:SS for sbatch/srun."
fi

# ====================================================================
# step 6: launcher
# ====================================================================
say "Step 6: launch method"
HAS_SBATCH=0; command -v sbatch >/dev/null 2>&1 && HAS_SBATCH=1
HAS_SRUN=0;   command -v srun   >/dev/null 2>&1 && HAS_SRUN=1
info "Detected: sbatch=$HAS_SBATCH srun=$HAS_SRUN enroot=$HAS_ENROOT docker=$NEED_DOCKER bare_ok=$BARE_OK login=$RUN_ON_LOGIN_NODE"

OPTS=(); KEYS=()
add_opt(){ OPTS+=("$1"); KEYS+=("$2"); }

if (( HAS_SBATCH == 1 && IS_CUSTOM == 0 )) && [[ -n "$CONT_REF" ]]; then
    add_opt "sbatch run.sub (Slurm+Pyxis+Enroot, containerized)" "sbatch"
fi
if (( HAS_SRUN == 1 && HAS_ENROOT == 1 && IS_CUSTOM == 0 )) && [[ -n "$CONT_REF" ]]; then
    add_opt "srun + Pyxis/Enroot (interactive)" "srun"
fi
if (( HAS_SBATCH == 1 && IS_CUSTOM == 0 && BARE_OK == 1 )); then
    add_opt "sbatch bare-metal (Slurm, no container)" "sbatch_bare"
fi
if (( HAS_SRUN == 1 && IS_CUSTOM == 0 && BARE_OK == 1 )); then
    add_opt "srun bare-metal (Slurm, no container)" "srun_bare"
fi
if (( RUN_ON_LOGIN_NODE == 0 )); then
    [[ -n "$IMAGE" ]] && add_opt "docker run (single-node)" "docker"
    (( BARE_OK == 1 )) && add_opt "bare-metal torchrun (single-node)" "bare"
    if (( IS_CUSTOM == 0 && BARE_OK == 1 )); then
        add_opt "bare-metal torchrun multi-node (no Slurm)" "bare_multi"
    fi
fi
if (( IS_CUSTOM == 1 )); then
    OPTS=(); KEYS=()
    [[ -n "$IMAGE" && $RUN_ON_LOGIN_NODE -eq 0 ]] && add_opt "docker smoke (recommended)" "smoke"
    # Bare smoke needs the full NeMo/Megatron/TE stack on the host python;
    # without a pre-built venv this means a multi-GB pip install. Offer it
    # but steer users toward docker smoke which already has everything.
    (( BARE_OK == 1 && RUN_ON_LOGIN_NODE == 0 )) \
        && add_opt "bare-metal smoke (requires NeMo+Megatron+TE on host)" "smoke_bare"
fi
add_opt "prepare-only (stop here; print commands)" "prepare"

(( ${#OPTS[@]} > 0 )) || die "No launch method available."
sel=$(pick "Choose launcher" "${OPTS[@]}")
METHOD="${KEYS[$((sel-1))]}"
# Only confirm method when the user actually had a choice. Single-option
# flows (e.g. only docker available) need no additional yes/no.
if (( ${#OPTS[@]} > 1 )); then
    yesno "Proceed with '$METHOD'?" y || die "Aborted."
fi

DEFAULT_MPORT=$(random_port)

docker_common_args=(
    --rm --gpus all --ipc=host --shm-size=16g --network=host
    --ulimit memlock=-1 --ulimit stack=67108864
    -v "$DATADIR/$WL_PREPROC_HOST_SUBPATH:$WL_PREPROC_MOUNT:ro"
    -v "$LOGDIR:/results"
    -e SEED="$SEED" -e WALLTIME="$WALLTIME"
    -e RANK=0 -e LOCAL_RANK=0 -e WORLD_SIZE=1 -e LOCAL_WORLD_SIZE=1
    -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT="$DEFAULT_MPORT"
    -e SLURM_JOB_ID=local -e SLURM_PROCID=0 -e SLURM_LOCALID=0
    # Cuts allocator fragmentation during smoke/reduced-layer runs where
    # activation footprint jumps step-to-step. Upstream prints this tip
    # on OOM; setting it pre-emptively avoids the retry cycle.
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
if [[ -n "$WL_TOKENIZER_HOST_SUBPATH" && -n "$WL_TOKENIZER_MOUNT" ]]; then
    docker_common_args+=(-v "$DATADIR/$WL_TOKENIZER_HOST_SUBPATH:$WL_TOKENIZER_MOUNT:ro")
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    docker_common_args+=(-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi
# Forward recipe overrides through to the container. Without this, the
# in-container `source $CFG_FILE` path ignores MAX_STEPS/LR/FP8/... that
# the dynamic menu computed on the host.
if (( ${REC_OVERRIDES_APPLIED:-0} == 1 )); then
    for _k in "${REC_EXPORT_VARS[@]}"; do
        [[ -n "${!_k:-}" ]] && docker_common_args+=(-e "$_k=${!_k}")
    done
fi
# Mount the picked config file into $WL_CONTAINER_WORKDIR so the in-container
# `source $CFG_FILE` resolves. Without this the host path is dereferenced
# inside the container where it does not exist (observed bug).
CFG_IN_CONTAINER=""
if [[ -n "$CFG_FILE" && -f "$CFG_FILE" ]]; then
    CFG_IN_CONTAINER="$WL_CONTAINER_WORKDIR/$(basename "$CFG_FILE")"
    docker_common_args+=(-v "$CFG_FILE:$CFG_IN_CONTAINER:ro")
fi

build_smoke_env_str() {
    local s=""
    local kv
    for kv in "${WL_SMOKE_ENV[@]}"; do s+=" $kv"; done
    local k
    for k in "${!SMOKE_PROMPT_VALUES[@]}"; do s+=" $k=${SMOKE_PROMPT_VALUES[$k]}"; done
    echo "$s"
}

# Emit sbatch-style comma-separated KEY=VAL list for recipe overrides so
# the Slurm launchers can pass them via --export=ALL,<list>. Empty when
# no recipe was selected.
build_recipe_export_list() {
    (( ${REC_OVERRIDES_APPLIED:-0} == 1 )) || return 0
    local list="" _k
    for _k in "${REC_EXPORT_VARS[@]}"; do
        [[ -n "${!_k:-}" ]] && list+=",${_k}=${!_k}"
    done
    echo "${list#,}"
}

bare_prereq_check() {
    say "Bare-metal prerequisite check"
    # Fallback hook: if bare prereqs fail but a docker image is cached,
    # give the user a one-keystroke switch instead of a dead-end die.
    _bare_fallback() {
        if [[ -n "${IMAGE:-}" ]] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
            warn "Bare-metal prereq: $1"
            if yesno "Switch to docker smoke using cached $IMAGE?" y; then
                if [[ "$METHOD" == "smoke_bare" ]]; then METHOD="smoke"
                elif [[ "$METHOD" == "bare" ]];       then METHOD="docker"
                fi
                info "Method switched: $METHOD"
                return 0
            fi
        fi
        die "$1"
    }
    if ! command -v python >/dev/null 2>&1; then _bare_fallback "python not on PATH"; return; fi
    if ! command -v torchrun >/dev/null 2>&1; then _bare_fallback "torchrun not on PATH"; return; fi
    info "python: $(python --version 2>&1)"
    python -c "import torch; print('torch=%s cuda=%s' % (torch.__version__, torch.cuda.is_available()))" \
        || { _bare_fallback "torch import failed on host python"; return; }
    # NeMo/Megatron inside the container are pinned to specific torch major.
    # A host torch built against a different CUDA runtime (e.g. host cu130 vs
    # container cu124) silently breaks TransformerEngine at runtime. Warn.
    local _host_torch; _host_torch=$(python -c "import torch,sys; sys.stdout.write(torch.__version__)" 2>/dev/null || echo "")
    if [[ -n "$_host_torch" && "$_host_torch" != 2.5* && "$_host_torch" != 2.6* ]]; then
        warn "Host torch=$_host_torch differs from container torch (upstream uses 2.5.x/2.6.x)."
        warn "NeMo/Megatron/TE may fail to import. Consider docker smoke instead."
        yesno "Continue anyway?" n || die "Aborted on torch-version mismatch."
    fi
    if [[ -f "$IMPL_DIR/requirements.txt" ]]; then
        if yesno "pip install -r $IMPL_DIR/requirements.txt?" n; then
            if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
                warn "No venv/conda active; refusing to pollute system site-packages."
                _VENV="$IMPL_DIR/.mlperf_venv"
                if yesno "Create and use $_VENV?" y; then
                    python -m venv "$_VENV" || die "venv create failed"
                    # shellcheck disable=SC1091
                    source "$_VENV/bin/activate"
                    info "venv active: $VIRTUAL_ENV"
                    pip install --upgrade pip wheel >/dev/null
                elif yesno "Fall back to 'pip install --user'?" n; then
                    pip install --user -r "$IMPL_DIR/requirements.txt" || warn "pip install issues"
                    return
                else
                    die "Cannot proceed without a pip target."
                fi
            fi
            pip install -r "$IMPL_DIR/requirements.txt" || warn "pip install issues"
        fi
    fi
}

write_host_env_snapshot() {
    # Bare-metal counterpart of run_and_time.sh's /results/container-env
    # dump — captures enough to reproduce the run env if it breaks later.
    {
        echo "# mlperf.sh host env snapshot — $(date -Iseconds)"
        echo "host: $(hostname)"
        echo "uname: $(uname -a)"
        echo "python: $(command -v python) $(python --version 2>&1)"
        echo "torchrun: $(command -v torchrun)"
        echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
        echo "--- env ---"
        env | grep -E '^(CUDA|NCCL|TORCH|MLPERF|OMP|PYTORCH)_' | sort
        echo "--- nvidia-smi ---"
        command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "(no nvidia-smi)"
    } > "$LOGDIR/host-env.log" 2>&1 || true
}

bare_env_export() {
    export DATADIR LOGDIR SEED
    : "${RANK:=0}"; : "${LOCAL_RANK:=0}"
    : "${WORLD_SIZE:=1}"; : "${LOCAL_WORLD_SIZE:=1}"
    : "${MASTER_ADDR:=127.0.0.1}"; : "${MASTER_PORT:=$DEFAULT_MPORT}"
    : "${SLURM_JOB_ID:=bare}"; : "${SLURM_PROCID:=$RANK}"; : "${SLURM_LOCALID:=$LOCAL_RANK}"
    : "${BINDCMD:=}"
    export RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
           SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID BINDCMD
}

emit_prepare_summary() {
    cat <<EOF

========= PREPARE-ONLY SUMMARY =========
workload     : $WL_NAME  —  $WL_DISPLAY
repo         : $REPO_DIR
impl dir     : $IMPL_DIR
image        : ${IMAGE:-<none>}
sqsh         : ${SQSH:-<none>}
cont ref     : ${CONT_REF:-<none>}
datadir      : $DATADIR/$WL_DATASET_SUBDIR
logdir       : $LOGDIR
config       : ${CFG_FILE:-<custom smoke>}
GPUs         : ${NGPU:-?} / ${GPU_TOTAL:-?}  (CVD=${CUDA_VISIBLE_DEVICES:-<all>})
nnodes       : ${DGXNNODES:-1}   gpus/node : ${DGXNGPU:-$NGPU}
seed         : $SEED   walltime: ${WALLTIME:-?}
ETA (total)  : $([[ -n "${EST_TOTAL_S:-}" && "${EST_TOTAL_S:-0}" -gt 0 ]] && _fmt_duration "$EST_TOTAL_S" || echo 'unknown')   verdict: ${EST_VERDICT:-unknown}
=========================================

To launch later:

# Slurm containerized (MLPerf-native):
CONT="$CONT_REF" DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \\
  sbatch -N ${DGXNNODES:-1} --time=${WALLTIME:-00:30:00} "$IMPL_DIR/run.sub"

# Single-node docker:
docker run --rm --gpus all --ipc=host --shm-size=16g \\
  -v "$DATADIR/$WL_PREPROC_HOST_SUBPATH:$WL_PREPROC_MOUNT:ro" \\
  -v "$LOGDIR:/results" \\
  ${IMAGE:-<image>} bash -c \\
    "cd $WL_CONTAINER_WORKDIR && source config_common.sh && source ${CFG_FILE:-<cfg>} && bash $WL_ENTRY"
EOF
}

# Re-source common configs and the picked config inside subshell to be safe.
source_configs() {
    set +u
    for f in config_common.sh config_common_cg.sh config_common_8b.sh; do
        [[ -f "$f" ]] && source "$f"
    done
    [[ -n "$CFG_FILE" && -f "$CFG_FILE" ]] && source "$CFG_FILE"
    # Recipe overrides must win over upstream `export MAX_STEPS=1200000`
    # etc. in config_common_*.sh. Re-assert here, last.
    if (( ${REC_OVERRIDES_APPLIED:-0} == 1 )); then
        local _k
        for _k in "${REC_EXPORT_VARS[@]}"; do
            [[ -n "${!_k:-}" ]] && export "$_k=${!_k}"
        done
    fi
    set -u
}

# --dry-run short-circuits to prepare-only for any launcher. Must fire
# before mkdir/preflight so exploratory runs leave no side effects.
if (( MLPERF_DRY_RUN == 1 )); then
    info "[dry-run] skipping launch; emitting prepare-only summary instead."
    METHOD=prepare
fi

# Run bare prereq check here (pre-dispatch) so its docker-fallback can
# actually reroute METHOD before the case below picks a branch.
case "$METHOD" in
    bare|bare_multi|smoke_bare) bare_prereq_check ;;
esac

# Pre-flight + LOGDIR mkdir only for paths that actually dispatch. Prepare
# mode should not touch disk.
preflight_mounts() {
    local missing=0 _p
    for _p in "$DATADIR/$WL_PREPROC_HOST_SUBPATH"; do
        [[ -d "$_p" ]] || { warn "Mount source missing: $_p"; missing=1; }
    done
    if [[ -n "$WL_TOKENIZER_HOST_SUBPATH" ]]; then
        [[ -d "$DATADIR/$WL_TOKENIZER_HOST_SUBPATH" ]] \
            || { warn "Tokenizer missing: $DATADIR/$WL_TOKENIZER_HOST_SUBPATH"; missing=1; }
    fi
    (( missing == 0 )) || die "Fix mounts above or run Step 3 download."
}
case "$METHOD" in
    smoke|docker|bare|bare_multi|smoke_bare)
        preflight_mounts
        mkdir -p "$LOGDIR"
        ;;
    sbatch|sbatch_bare|srun|srun_bare)
        # Cluster launchers write to $LOGDIR from the compute node — create
        # it on the submit host so Slurm --output has somewhere to land.
        mkdir -p "$LOGDIR"
        ;;
esac

# Pre-launch confirmation banner. User sees ETA + verdict one more time
# before real side-effects start. `prepare` skips (it's already summary-only).
if [[ "$METHOD" != "prepare" ]]; then
    say "Launch preview"
    info "workload : $WL_NAME  on ${NGPU:-?}×sm_${AUTO_GPU_ARCH:-?}  method=$METHOD"
    info "config   : $(basename "${CFG_FILE:-<custom smoke>}")"
    info "ETA      : $([[ -n "${EST_TOTAL_S:-}" && "${EST_TOTAL_S:-0}" -gt 0 ]] && _fmt_duration "$EST_TOTAL_S" || echo 'unknown')   verdict: ${EST_VERDICT:-unknown}"
    case "${EST_VERDICT:-}" in
        non-compliant|infeasible)
            warn "This run will NOT produce a submittable MLPerf result."
            yesno "Proceed anyway?" n || die "Aborted by user." ;;
        unknown)
            warn "No step-time data for this (workload,arch,world). ETA unavailable; run will populate cache."
            yesno "Proceed?" y || die "Aborted by user." ;;
        *)
            yesno "Proceed with launch?" y || die "Aborted by user." ;;
    esac
fi

# Live monitor — starts now, observes $LOGDIR MLlog until parent exits or
# mon_stop is called. Only active for in-process launchers (docker/bare/
# smoke); sbatch/srun run on remote nodes where this monitor can't see logs.
_MON_BASELINE=0
if type rec_best_probe >/dev/null 2>&1 \
   && [[ -n "${REC_CHOICE_NAME:-}" && "$REC_CHOICE_NAME" != "custom" ]]; then
    _probe_line=$(rec_best_probe "$WL_NAME" "$NGPU" "${REC_CHOICE_FP8:-0}") || true
    [[ -n "$_probe_line" ]] && _MON_BASELINE=$(awk '{print $5}' <<<"$_probe_line")
fi
case "$METHOD" in
    docker|bare|bare_multi|smoke|smoke_bare)
        if type mon_start >/dev/null 2>&1; then
            mkdir -p "$LOGDIR"
            mon_start "$LOGDIR" "$$" "$_MON_BASELINE" || true
            # Compose with the existing cleanup trap (line 142). Setting a
            # new trap replaces the previous one in bash, so we must call
            # both cleanup (container kill) and mon_stop here.
            trap 'cleanup; type mon_stop >/dev/null 2>&1 && mon_stop "$LOGDIR"' EXIT
        fi
        ;;
esac

case "$METHOD" in
    prepare)
        emit_prepare_summary
        exit 0
        ;;
    sbatch)
        [[ -n "$CONT_REF" ]] \
            || die "sbatch needs a container ref (CONT_REF). Re-run Step 2 and provide a pyxis-style ref (docker://... or /path/to/image.sqsh)."
        ACCOUNT="$(ask 'Slurm --account (blank to skip)' "${AUTO_SLURM_ACCOUNT:-}")"
        PARTITION="$(ask 'Slurm --partition (blank to skip)' "${AUTO_SLURM_PARTITION:-}")"
        RESERVATION="$(ask 'Slurm --reservation (blank to skip)' '')"
        NEXP="$(ask 'NEXP' 1)"
        ARGS=(-N "$DGXNNODES" --time="$WALLTIME")
        [[ -n "$ACCOUNT" ]]     && ARGS+=(--account="$ACCOUNT")
        [[ -n "$PARTITION" ]]   && ARGS+=(--partition="$PARTITION")
        [[ -n "$RESERVATION" ]] && ARGS+=(--reservation="$RESERVATION")
        _rec_export=$(build_recipe_export_list)
        [[ -n "$_rec_export" ]] && ARGS+=(--export="ALL,$_rec_export")
        CONT="$CONT_REF" DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" NEXP="$NEXP" \
            sbatch "${ARGS[@]}" run.sub
        ;;
    srun)
        [[ -n "$CONT_REF" ]] \
            || die "srun needs a container ref (CONT_REF). Re-run Step 2 and provide a pyxis-style ref."
        NODES="$(ask 'Nodes (-N)' "$DGXNNODES")"
        NTPN="$(ask 'ntasks-per-node' "$DGXNGPU")"
        GPUS="$(ask 'gpus-per-task' 1)"
        TIME="$(ask 'time limit' "$WALLTIME")"
        mounts="$DATADIR/$WL_PREPROC_HOST_SUBPATH:$WL_PREPROC_MOUNT:ro,$LOGDIR:/results"
        [[ -n "$WL_TOKENIZER_HOST_SUBPATH" ]] && \
            mounts="$mounts,$DATADIR/$WL_TOKENIZER_HOST_SUBPATH:$WL_TOKENIZER_MOUNT:ro"
        CONT="$CONT_REF" DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \
            srun -N "$NODES" --ntasks-per-node="$NTPN" --gpus-per-task="$GPUS" --time="$TIME" \
                 --container-image="$CONT_REF" --container-mounts="$mounts" \
                 --container-workdir="$WL_CONTAINER_WORKDIR" \
                 bash "$WL_ENTRY"
        ;;
    sbatch_bare)
        ACCOUNT="$(ask 'Slurm --account (blank to skip)' "${AUTO_SLURM_ACCOUNT:-}")"
        PARTITION="$(ask 'Slurm --partition (blank to skip)' "${AUTO_SLURM_PARTITION:-}")"
        WRAP="cd '$IMPL_DIR' && source config_common.sh && [[ -f config_common_cg.sh ]] && source config_common_cg.sh; [[ -f config_common_8b.sh ]] && source config_common_8b.sh; source '$CFG_FILE' && bash $WL_ENTRY"
        ARGS=(-N "$DGXNNODES" --ntasks-per-node="$DGXNGPU" --gpus-per-task=1 --time="$WALLTIME")
        [[ -n "$ACCOUNT" ]]   && ARGS+=(--account="$ACCOUNT")
        [[ -n "$PARTITION" ]] && ARGS+=(--partition="$PARTITION")
        _rec_export=$(build_recipe_export_list)
        [[ -n "$_rec_export" ]] && ARGS+=(--export="ALL,$_rec_export")
        DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \
            sbatch "${ARGS[@]}" --wrap="$WRAP"
        ;;
    srun_bare)
        NODES="$(ask 'Nodes (-N)' "$DGXNNODES")"
        NTPN="$(ask 'ntasks-per-node' "$DGXNGPU")"
        GPUS="$(ask 'gpus-per-task' 1)"
        TIME="$(ask 'time limit' "$WALLTIME")"
        DATADIR="$DATADIR" LOGDIR="$LOGDIR" SEED="$SEED" \
            srun -N "$NODES" --ntasks-per-node="$NTPN" --gpus-per-task="$GPUS" --time="$TIME" \
                 --chdir="$IMPL_DIR" \
                 bash -c "source config_common.sh && source '$CFG_FILE' && bash $WL_ENTRY"
        ;;
    docker)
        CNAME="mlperf-$WL_NAME-$$-$(date +%s)"
        track_container "$CNAME"
        # When a recipe is active, write its overrides to a file mounted
        # into the container, sourced AFTER config_common*.sh / CFG_FILE.
        # Otherwise upstream `export MAX_STEPS=1200000` clobbers the -e
        # vars and silently discards the recipe.
        RECIPE_OVERRIDE_SRC=""
        if (( ${REC_OVERRIDES_APPLIED:-0} == 1 )); then
            _rof="$LOGDIR/recipe_overrides.env"
            : > "$_rof"
            for _k in "${REC_EXPORT_VARS[@]}"; do
                [[ -n "${!_k:-}" ]] && printf 'export %s=%q\n' "$_k" "${!_k}" >> "$_rof"
            done
            docker_common_args+=(-v "$_rof:/tmp/recipe_overrides.env:ro")
            RECIPE_OVERRIDE_SRC='source /tmp/recipe_overrides.env'
            info "Recipe overrides staged at $_rof (sourced last inside container)."
        fi
        docker run --name "$CNAME" "${docker_common_args[@]}" \
            "$IMAGE" bash -c "
                set -e
                cd $WL_CONTAINER_WORKDIR
                [ -f config_common.sh ]    && source config_common.sh    || true
                [ -f config_common_cg.sh ] && source config_common_cg.sh || true
                [ -f config_common_8b.sh ] && source config_common_8b.sh || true
                source ${CFG_IN_CONTAINER:-$CFG_FILE}
                export DGXNGPU=$NGPU DGXNNODES=1 BINDCMD=''
                $RECIPE_OVERRIDE_SRC
                bash $WL_ENTRY
            "
        ;;
    bare)
        bare_env_export
        write_host_env_snapshot
        [[ -n "$WL_TOKENIZER_HOST_SUBPATH" && -n "$WL_TOKENIZER_MOUNT" ]] && \
            ln -sfn "$DATADIR/$WL_TOKENIZER_HOST_SUBPATH" "$IMPL_DIR/$(basename "$WL_TOKENIZER_MOUNT")"
        # Same /results redirect as smoke_bare — run_and_time.sh writes
        # container-env-<jobid>.log to /results even on the bare path.
        PATCHED_ENTRY="$LOGDIR/${WL_ENTRY}.patched"
        sed -e "s|/results|$LOGDIR|g" "$IMPL_DIR/$WL_ENTRY" > "$PATCHED_ENTRY"
        chmod +x "$PATCHED_ENTRY"
        (
            cd "$IMPL_DIR" || exit 1
            source_configs
            export DGXNGPU="$NGPU" DGXNNODES=1 BINDCMD=""
            bash "$PATCHED_ENTRY"
        )
        ;;
    smoke)
        CNAME="mlperf-$WL_NAME-smoke-$$-$(date +%s)"
        track_container "$CNAME"
        SMOKE_STR="$(build_smoke_env_str)"
        # Upstream Llama31Config8B hard-codes num_layers=32 even when
        # OVERWRITTEN_NUM_LAYERS is set (env var is only honored for 405b).
        # Inject a sitecustomize.py via PYTHONPATH so the env truncates 8B
        # too — required for smoke to fit on a single H200 (140 GB).
        docker run --name "$CNAME" "${docker_common_args[@]}" \
            -e SLURM_JOB_ID=smoke \
            "$IMAGE" bash -c "
                set -e
                mkdir -p /tmp/mlperf_smoke_patch
                cat > /tmp/mlperf_smoke_patch/sitecustomize.py <<'PY'
import os
try:
    from nemo.collections.llm.gpt.model.llama import (
        Llama31Config8B, Llama31Config70B, Llama31Config405B)
    n = int(os.environ.get('OVERWRITTEN_NUM_LAYERS') or 0)
    if n > 0:
        for cls in (Llama31Config8B, Llama31Config70B, Llama31Config405B):
            _orig = cls.__init__
            def _mk(orig, nn):
                def __init__(self, *a, **k):
                    orig(self, *a, **k); self.num_layers = nn
                return __init__
            cls.__init__ = _mk(_orig, n)
except Exception as e:
    import sys; print('smoke monkey-patch skipped:', e, file=sys.stderr)
PY
                export PYTHONPATH=/tmp/mlperf_smoke_patch:\${PYTHONPATH:-}
                cd $WL_CONTAINER_WORKDIR
                [ -f config_common.sh ]    && source config_common.sh    || true
                [ -f config_common_cg.sh ] && source config_common_cg.sh || true
                [ -f config_common_8b.sh ] && source config_common_8b.sh || true
                export $SMOKE_STR
                bash $WL_ENTRY
            "
        ;;
    smoke_bare)
        bare_env_export
        write_host_env_snapshot
        [[ -n "$WL_TOKENIZER_HOST_SUBPATH" && -n "$WL_TOKENIZER_MOUNT" ]] && \
            ln -sfn "$DATADIR/$WL_TOKENIZER_HOST_SUBPATH" "$IMPL_DIR/$(basename "$WL_TOKENIZER_MOUNT")"
        # Bare-metal has no /results mount — run_and_time.sh writes
        # container-env-*.log there. Patch a local copy to redirect.
        mkdir -p "$LOGDIR"
        PATCHED_ENTRY="$LOGDIR/${WL_ENTRY}.patched"
        sed -e "s|/results|$LOGDIR|g" "$IMPL_DIR/$WL_ENTRY" > "$PATCHED_ENTRY"
        chmod +x "$PATCHED_ENTRY"
        # Inject sitecustomize.py for 8B num_layers override.
        SMOKE_PATCH_DIR="$LOGDIR/smoke_patch"
        mkdir -p "$SMOKE_PATCH_DIR"
        cat > "$SMOKE_PATCH_DIR/sitecustomize.py" <<'PY'
import os
try:
    from nemo.collections.llm.gpt.model.llama import (
        Llama31Config8B, Llama31Config70B, Llama31Config405B)
    n = int(os.environ.get('OVERWRITTEN_NUM_LAYERS') or 0)
    if n > 0:
        for cls in (Llama31Config8B, Llama31Config70B, Llama31Config405B):
            _orig = cls.__init__
            def _mk(orig, nn):
                def __init__(self, *a, **k):
                    orig(self, *a, **k); self.num_layers = nn
                return __init__
            cls.__init__ = _mk(_orig, n)
except Exception as e:
    import sys; print('smoke monkey-patch skipped:', e, file=sys.stderr)
PY
        export PYTHONPATH="$SMOKE_PATCH_DIR:${PYTHONPATH:-}"
        (
            cd "$IMPL_DIR" || exit 1
            source_configs
            export "${WL_SMOKE_ENV[@]}"
            for k in "${!SMOKE_PROMPT_VALUES[@]}"; do export "$k=${SMOKE_PROMPT_VALUES[$k]}"; done
            bash "$PATCHED_ENTRY"
        )
        ;;
    bare_multi)
        [[ -n "$WL_PRETRAIN_PY" ]] || die "bare_multi requires WL_PRETRAIN_PY in manifest."
        NNODES="$(ask 'Total NNODES' "$DGXNNODES")"
        NODE_RANK="$(ask_req 'This NODE_RANK')"
        GPUS_PER_NODE="$(ask 'GPUs per node' "$NGPU")"
        M_ADDR="$(ask_req 'MASTER_ADDR')"
        M_PORT="$(ask 'MASTER_PORT' "$DEFAULT_MPORT")"
        RDZV_ID="$(ask 'rendezvous id (same on all nodes)' "mlperf-$WL_NAME")"
        [[ -n "$WL_TOKENIZER_HOST_SUBPATH" && -n "$WL_TOKENIZER_MOUNT" ]] && \
            ln -sfn "$DATADIR/$WL_TOKENIZER_HOST_SUBPATH" "$IMPL_DIR/$(basename "$WL_TOKENIZER_MOUNT")"
        export DATADIR LOGDIR SEED
        (
            cd "$IMPL_DIR" || exit 1
            source_configs
            export DGXNGPU="$GPUS_PER_NODE" DGXNNODES="$NNODES" BINDCMD=""
            torchrun \
                --nnodes="$NNODES" --node_rank="$NODE_RANK" \
                --nproc_per_node="$GPUS_PER_NODE" \
                --rdzv_id="$RDZV_ID" --rdzv_backend=c10d \
                --rdzv_endpoint="$M_ADDR:$M_PORT" \
                "$WL_PRETRAIN_PY"
        )
        ;;
    *) die "Unknown method: $METHOD" ;;
esac

ec=$?
# Stop live monitor (no-op if never started) and print alert summary.
if type mon_stop >/dev/null 2>&1; then
    mon_stop "$LOGDIR" 2>/dev/null || true
    type mon_report >/dev/null 2>&1 && mon_report "$LOGDIR" || true
fi
# After run, record measured step-time into calibration cache so future
# ETAs and recipe menus reflect reality on this hardware.
if type cal_cache_record >/dev/null 2>&1 && [[ -d "$LOGDIR" ]]; then
    _post_step=$(grep -rh '"train_step_time"' "$LOGDIR" 2>/dev/null \
               | sed -nE 's/.*"train_step_time"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p' \
               | awk 'NR>10' | sort -g \
               | awk '{a[NR]=$1} END{if(NR==0) exit; if(NR%2) print a[(NR+1)/2]; else printf "%.6f\n",(a[NR/2]+a[NR/2+1])/2}')
    if [[ -n "$_post_step" ]]; then
        cal_inventory_hw
        # Normalize by MICRO_BATCH so cache key matches the probe (which
        # always runs at microbs=1). Step-time scales linearly with
        # microbs at steady state, so division gives a comparable value.
        _mbs="${MICRO_BATCH_SIZE:-1}"; (( _mbs < 1 )) && _mbs=1
        _norm_step=$(awk -v s="$_post_step" -v m="$_mbs" 'BEGIN{printf "%.6f", s/m}')
        cal_cache_record "$WL_NAME" "$NGPU" \
            "${TENSOR_MODEL_PARALLEL:-1}" "${PIPELINE_MODEL_PARALLEL:-1}" \
            "${CONTEXT_PARALLEL:-1}" "$([[ "${FP8:-False}" == "True" ]] && echo 1 || echo 0)" \
            1 "$_norm_step" 0 0
        info "Cached measured step-time ${_post_step}s (normalized ${_norm_step}s/microbs=1) → $CAL_CACHE"
    fi
fi
if (( ec == 0 )); then
    say "Done. Outputs: $LOGDIR"
    # Persist answers so next run defaults to what worked this run. Save
    # LOGDIR_PARENT (user-typed) as LOGDIR_PREV — we re-timestamp every run.
    LOGDIR_PREV="${LOGDIR_PARENT:-$(dirname "$LOGDIR")}"
    state_save WL_NAME REPO_DIR DATADIR LOGDIR_PREV SEED IMAGE CONT_REF \
               METHOD CFG_FILE 2>/dev/null || true
    # Pointer to the latest successful run for quick `cd`.
    ln -sfn "$LOGDIR" "$(dirname "$LOGDIR")/latest" 2>/dev/null || true

    # Post-preview pointer: if this was Smoke/Shape (tiny step count),
    # the run only proved the pipeline works. Tell the user how to
    # kick off a real run now that the calibration cache is warm.
    if (( MLPERF_AUTO_YES == 0 )) \
       && [[ "${REC_CHOICE_NAME:-}" == "Smoke" || "${REC_CHOICE_NAME:-}" == "Shape check" ]]; then
        say "Preview ok. Pipeline proven. To start a real run:"
        info "  bash \"$0\""
        info "  # calibration cache already warm — recipe menu appears instantly."
        info "  # Recommended picks:"
        info "  #   [3] Short convergence — engineering run, loss descends"
        info "  #   [4] Full convergence  — MLPerf target (WARN: multi-day on small clusters)"
        info "  #   [5] FP8 throughput    — benchmark FP8 vs BF16"
        info "Full-run ETAs were displayed in the recipe menu earlier this session."
    fi
    case "$METHOD" in
        sbatch|sbatch_bare)
            info "For closed-division compliance, validate with:"
            info "  bash \"$SCRIPT_DIR/tools/compliance.sh\"  # point at $LOGDIR"
            ;;
        docker|bare|bare_multi|smoke|smoke_bare)
            info "Run was NOT launched via sbatch run.sub — not MLPerf-compliant."
            info "For a compliant run use the sbatch launcher with --container-image."
            _record_step_time_from_log "$LOGDIR" "$WL_NAME" "${AUTO_GPU_ARCH:-0}" "${NGPU:-1}"
            ;;
    esac
else
    err "Exit code $ec."
    # Post-mortem: look for stderr/output inside LOGDIR before telling the
    # user "see $LOGDIR" — the log showed empty dirs left behind with no
    # pointer to what actually went wrong.
    if [[ -d "$LOGDIR" ]]; then
        _files=$(find "$LOGDIR" -maxdepth 2 -type f 2>/dev/null | head -5)
        if [[ -z "$_files" ]]; then
            warn "LOGDIR $LOGDIR is empty — process died before writing output."
            warn "Likely cause: prereq/mount/permission issue printed above."
            # Clean up the empty timestamped dir so results/ doesn't clutter.
            rmdir "$LOGDIR" 2>/dev/null && info "Removed empty LOGDIR."
        else
            info "Last output under $LOGDIR:"
            for _f in $_files; do info "  $_f"; done
            # Tail the freshest file — often the only hint of the failure.
            _latest=$(ls -t $_files 2>/dev/null | head -1)
            if [[ -n "$_latest" ]]; then
                warn "Tail of $_latest:"
                tail -n 20 "$_latest" 2>/dev/null | sed 's/^/    /' >&2
            fi
        fi
    fi
fi
exit "$ec"
