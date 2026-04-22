# shellcheck shell=bash
# Common helpers for mlperf.sh and tools/*.sh.
# Source (do not execute) from a script that sets MLPERF_LIB_MODE=1.
#
#     : "${MLPERF_LIB_MODE:=1}"
#     # shellcheck source=../lib/common.sh
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"

# Guard against double-sourcing.
[[ -n "${_MLPERF_COMMON_LOADED:-}" ]] && return 0
_MLPERF_COMMON_LOADED=1

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
say()  { printf "\n==> %s\n" "$*"; }
info() { printf "    %s\n" "$*"; }
warn() { printf "WARN: %s\n" "$*" >&2; }
err()  { printf "ERROR: %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

# Global flags — scripts honour these after calling parse_common_flags.
MLPERF_AUTO_YES=0
MLPERF_DRY_RUN=0
MLPERF_CONFIG_FILE=""

# Parse --yes / --dry-run / --config FILE out of "$@". Remaining args are
# re-assigned via `eval "set -- \"\${REMAINING[@]}\""` by the caller.
parse_common_flags() {
    REMAINING=()
    while (( $# )); do
        case "$1" in
            --yes|-y)   MLPERF_AUTO_YES=1 ;;
            --dry-run)  MLPERF_DRY_RUN=1 ;;
            --config)   shift; MLPERF_CONFIG_FILE="${1:-}" ;;
            --config=*) MLPERF_CONFIG_FILE="${1#*=}" ;;
            --help|-h)  MLPERF_PRINT_HELP=1 ;;
            *)          REMAINING+=("$1") ;;
        esac
        shift
    done
    if [[ -n "$MLPERF_CONFIG_FILE" ]]; then
        [[ -f "$MLPERF_CONFIG_FILE" ]] || die "Config file not found: $MLPERF_CONFIG_FILE"
        # shellcheck disable=SC1090
        source "$MLPERF_CONFIG_FILE"
        MLPERF_AUTO_YES=1
        info "Loaded config: $MLPERF_CONFIG_FILE (auto-yes enabled)"
    fi
}

# ask "prompt" "default" — returns default verbatim when auto-yes active.
ask() {
    local p="$1" d="${2-}" v=""
    if (( MLPERF_AUTO_YES == 1 )); then echo "$d"; return; fi
    if [[ -n "$d" ]]; then read -r -p "$p [$d]: " v; echo "${v:-$d}"
    else                   read -r -p "$p: "        v; echo "$v"; fi
}

ask_req() {
    local p="$1" v=""
    if (( MLPERF_AUTO_YES == 1 )); then die "required value '$p' not supplied in config"; fi
    while :; do read -r -p "$p: " v; [[ -n "$v" ]] && { echo "$v"; return; }; err "value required"; done
}

yesno() {
    local p="$1" d="${2-y}" v=""
    if (( MLPERF_AUTO_YES == 1 )); then [[ "$d" == "y" ]]; return; fi
    while :; do read -r -p "$p (y/n) [$d]: " v; v="${v:-$d}"
        case "$v" in [Yy]|[Yy][Ee][Ss]) return 0;;
                     [Nn]|[Nn][Oo])      return 1;;
                     *) err "Answer y or n";; esac
    done
}

pick() {
    local p="$1"; shift
    local i=1; for o in "$@"; do printf "  [%d] %s\n" "$i" "$o" >&2; i=$((i+1)); done
    if (( MLPERF_AUTO_YES == 1 )); then echo 1; return; fi
    local v=""
    while :; do read -r -p "$p [1]: " v; v="${v:-1}"
        [[ "$v" =~ ^[0-9]+$ ]] && (( v>=1 && v<=$# )) && { echo "$v"; return; }
        err "Enter 1..$#"
    done
}

# ----------------------------------------------------------------------
# Path validation
# ----------------------------------------------------------------------
validate_path() {
    local p="$1" label="$2"
    if [[ "$p" =~ [[:space:]] ]]; then die "$label path must not contain spaces: '$p'"; fi
    if [[ "$p" =~ [,\;\|\&\$\`\"\'\(\)] ]]; then die "$label path has shell-special chars: '$p'"; fi
}

# ----------------------------------------------------------------------
# Retry wrapper (exponential backoff, 3 attempts default)
# ----------------------------------------------------------------------
retry() {
    local n="${MLPERF_RETRY_TRIES:-3}" delay="${MLPERF_RETRY_DELAY:-5}" attempt=1
    while (( attempt <= n )); do
        if "$@"; then return 0; fi
        (( attempt < n )) && warn "retry $attempt/$((n-1)) failed for: $* — sleeping ${delay}s"
        sleep "$delay"
        delay=$((delay * 2))
        attempt=$((attempt + 1))
    done
    err "giving up after $n attempts: $*"
    return 1
}

# ----------------------------------------------------------------------
# Resume/state persistence
# ----------------------------------------------------------------------
: "${MLPERF_STATE_DIR:=${XDG_STATE_HOME:-$HOME/.local/state}/mlperf}"
state_save() {
    mkdir -p "$MLPERF_STATE_DIR"
    { for v in "$@"; do declare -p "$v" 2>/dev/null; done; } > "$MLPERF_STATE_DIR/session.env"
}
state_load() {
    [[ -f "$MLPERF_STATE_DIR/session.env" ]] || return 1
    # shellcheck disable=SC1091
    source "$MLPERF_STATE_DIR/session.env"
}

# ----------------------------------------------------------------------
# Notification hook (Slack / webhook / noop)
# ----------------------------------------------------------------------
notify() {
    local msg="$1" status="${2:-info}"
    [[ -z "${MLPERF_NOTIFY_URL:-}" ]] && return 0
    command -v curl >/dev/null 2>&1 || return 0
    local payload; payload=$(printf '{"text":"[mlperf] %s: %s"}' "$status" "$msg")
    curl -fsS -X POST -H 'Content-Type: application/json' \
        --data "$payload" "$MLPERF_NOTIFY_URL" >/dev/null 2>&1 || true
}

# ----------------------------------------------------------------------
# Platform detect
# ----------------------------------------------------------------------
detect_platform() {
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
    [[ "$PLATFORM" == "windows" ]] && export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
}

# ----------------------------------------------------------------------
# Pinned-version manifest (change here, not in N scripts)
# ----------------------------------------------------------------------
PIN_MLPERF_LOGGING="3.1.0"
PIN_PYXIS_SHA="5fa3c38c73aab30adb9f7a1ff3c37b89d0938a43"   # v0.20.0
PIN_ENROOT_VERSION="3.5.0"
PIN_SLURM_PACKAGE_NOTE="distro (override via upstream repo)"
