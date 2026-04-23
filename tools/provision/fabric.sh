#!/usr/bin/env bash
# Fabric manager + Mellanox/NVIDIA OFED + RDMA tuning for multi-node MLPerf.
#
# Covers:
#   - nvidia-fabricmanager service (required on GB200/GB300 with NVSwitch)
#   - MLNX_OFED / DOCA-OFED or NVIDIA OFED drivers (one of three paths)
#   - rdma-core from distro (fallback when vendor OFED is not allowed)
#   - Performance sysctls for InfiniBand + RoCE
#   - IB perftest sanity probe (ib_read_bw, ib_write_lat)
#
# Idempotent on re-run. Must be run as root (or passwordless sudo).
# Assumes bootstrap_node.sh has already been applied.

set -u
set -o pipefail

# --- mlperf.sh common-lib hook -----------------------------------------
_MLPERF_LIB_SOURCED=0
if _LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/common.sh" && [[ -f "$_LIB" ]]; then
    # shellcheck source=../lib/common.sh
    source "$_LIB"
    _MLPERF_LIB_SOURCED=1
fi
# Auto-yes / config-file via env only — no flag parsing here to avoid
# clobbering per-tool argv handling.
: "${MLPERF_AUTO_YES:=0}"
if [[ -n "${MLPERF_CONFIG_FILE:-}" && -f "${MLPERF_CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${MLPERF_CONFIG_FILE}"
    MLPERF_AUTO_YES=1
fi
# -----------------------------------------------------------------------

CUDA_VERSION_SUFFIX="${CUDA_VERSION_SUFFIX:-12-4}"   # for nvidia-fabricmanager-<cuda>
OFED_URL_DEFAULT="https://content.mellanox.com/ofed/MLNX_OFED-LATEST/MLNX_OFED_LINUX.tgz"
DOCA_HOST_URL_DEFAULT="https://developer.nvidia.com/doca-download"

[[ $EUID -eq 0 ]] || { command -v sudo >/dev/null || die "Need root or sudo"; SUDO=sudo; }
: "${SUDO:=}"

OS_ID=$(. /etc/os-release; echo "$ID")
OS_VER=$(. /etc/os-release; echo "$VERSION_ID")
case "$OS_ID" in
    ubuntu) PKG=apt-get ;;
    rhel|centos|rocky|almalinux) PKG=dnf ;;
    *) die "Unsupported OS: $OS_ID" ;;
esac
pkg_install(){ if [[ "$PKG" == "apt-get" ]]; then DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "$@"; else $SUDO $PKG install -y "$@"; fi; }
pkg_update(){ if [[ "$PKG" == "apt-get" ]]; then $SUDO apt-get update; else $SUDO $PKG makecache -y; fi; }

# ============================================================
# Step 1 — nvidia-fabricmanager  (only on systems with NVSwitch)
# ============================================================
say "Step 1: NVIDIA Fabric Manager"
HAS_NVSWITCH=0
if lspci 2>/dev/null | grep -qi nvidia | grep -qi switch; then HAS_NVSWITCH=1; fi
[[ $HAS_NVSWITCH == 0 ]] && command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi nvlink --status 2>/dev/null | grep -q NVLink && HAS_NVSWITCH=1
if (( HAS_NVSWITCH == 1 )); then
    if systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
        info "nvidia-fabricmanager already running."
    else
        if [[ "$PKG" == "apt-get" ]]; then
            # APT packaging: nvidia-fabricmanager-<cuda-major-minor>
            pkg_update
            pkg_install "nvidia-fabricmanager-$CUDA_VERSION_SUFFIX" || die "FM install failed"
        else
            # dnf/yum packaging ships either nvidia-fabric-manager (driver repo)
            # or cuda-drivers-fabricmanager-<cuda> on newer repos. Try both.
            pkg_install "cuda-drivers-fabricmanager-$CUDA_VERSION_SUFFIX" \
                || pkg_install nvidia-fabric-manager \
                || die "FM install failed (tried cuda-drivers-fabricmanager-$CUDA_VERSION_SUFFIX and nvidia-fabric-manager)"
        fi
        $SUDO systemctl enable --now nvidia-fabricmanager
        info "Started nvidia-fabricmanager."
    fi
else
    info "No NVSwitch detected; skipping fabric-manager (expected for single-GPU nodes, H100 NVLink-only pairs)."
fi

# ============================================================
# Step 2 — OFED driver stack selection
# ============================================================
say "Step 2: OFED / RDMA stack"
if command -v ofed_info >/dev/null 2>&1; then
    info "Vendor OFED already installed: $(ofed_info -s 2>/dev/null | head -1)"
    SKIP_OFED=1
else
    SKIP_OFED=0
    sel=$(pick "Choose RDMA stack" \
        "MLNX_OFED (legacy name, recommended for InfiniBand)" \
        "NVIDIA DOCA-Host OFED (newer, required on ConnectX-7/BlueField-3)" \
        "Distro rdma-core only (no vendor OFED; limited feature set)" \
        "skip — already handled by site image")
    case "$sel" in
        1)
            URL="${MLNX_OFED_URL:-$OFED_URL_DEFAULT}"
            TMP=$(mktemp -d)
            info "Downloading MLNX_OFED from $URL"
            curl -fsSL -o "$TMP/ofed.tgz" "$URL" || die "Download failed; set MLNX_OFED_URL."
            tar -xzf "$TMP/ofed.tgz" -C "$TMP"
            INSTALLER="$(find "$TMP" -maxdepth 2 -name mlnxofedinstall -print -quit)"
            [[ -n "$INSTALLER" ]] || die "mlnxofedinstall not found in archive."
            if yesno "Run '$INSTALLER --force' now? (rebuilds initramfs, ~10 min)" y; then
                if ! $SUDO "$INSTALLER" --force; then
                    rm -rf "$TMP"
                    die "mlnxofedinstall failed"
                fi
            else
                info "Skipped. Installer at $INSTALLER"
            fi
            rm -rf "$TMP"
            ;;
        2)
            info "DOCA-Host requires interactive download from $DOCA_HOST_URL_DEFAULT"
            info "After downloading, run: sudo ./doca-host-repo-*.sh --install"
            ;;
        3)
            pkg_install rdma-core libibverbs-utils ibverbs-providers \
                        infiniband-diags perftest librdmacm1 librdmacm-dev
            ;;
        4) : ;;
    esac
fi

# ============================================================
# Step 3 — RDMA / IB sysctls and udev
# ============================================================
say "Step 3: IB / RoCE tuning"
cat <<'EOF' | $SUDO tee /etc/sysctl.d/99-mlperf-fabric.conf >/dev/null
# Tune TCP for RoCE v2 with jumbo MTU.
net.core.rmem_max = 2147483647
net.core.wmem_max = 2147483647
net.core.rmem_default = 212992
net.core.wmem_default = 212992
net.ipv4.tcp_rmem = 4096 87380 2147483647
net.ipv4.tcp_wmem = 4096 65536 2147483647
net.ipv4.tcp_mem = 786432 1048576 26777216
# UDP for GPUDirect RDMA.
net.core.netdev_budget = 65535
net.core.netdev_max_backlog = 250000
net.core.somaxconn = 65535
# Disable reverse-path filtering on RoCE/IB interfaces (PFC).
net.ipv4.conf.all.rp_filter = 0
net.ipv4.conf.default.rp_filter = 0
EOF
$SUDO sysctl -p /etc/sysctl.d/99-mlperf-fabric.conf || true

# GPUDirect RDMA kernel module (nvidia_peermem). Needed for NCCL to use IB.
if ! lsmod | grep -q nvidia_peermem; then
    info "Loading nvidia_peermem (GPUDirect RDMA)"
    $SUDO modprobe nvidia_peermem 2>/dev/null \
        || warn "nvidia_peermem module not available; GPUDirect RDMA disabled."
fi
echo 'nvidia_peermem' | $SUDO tee /etc/modules-load.d/nvidia_peermem.conf >/dev/null

# ============================================================
# Step 4 — Sanity probes
# ============================================================
say "Step 4: Sanity probes"
if command -v ibstat >/dev/null 2>&1; then
    info "ibstat:"
    ibstat | awk 'NR <= 25'
else
    warn "ibstat missing — InfiniBand user-space not installed."
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    info "GPU topology (nvidia-smi topo -m):"
    nvidia-smi topo -m 2>/dev/null | head -30 || true
fi
if command -v ib_read_bw >/dev/null 2>&1; then
    info "Run (on two nodes) to validate IB fabric:"
    info "  host A:  ib_read_bw  -d mlx5_0  -a  -F"
    info "  host B:  ib_read_bw  -d mlx5_0  -a  -F  <host-A-IP>"
fi

say "Fabric provisioning complete."
info "If NVSwitch is present, reboot once to load the FM's updated DCGM state."
info "Validate multi-node NCCL: srun --container-image=... nccl_all_reduce_perf"
