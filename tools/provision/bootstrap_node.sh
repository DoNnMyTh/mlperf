#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 22.04 / 24.04 node (or RHEL 9) for MLPerf training.
#
# Installs, in order: kernel headers + build tools; NVIDIA data-centre driver
# from CUDA repo; nvidia-container-toolkit; Docker CE; Enroot; Pyxis SPANK
# plugin; Munge; Slurm; optional MIG partitioning helper; hugepages + RDMA
# sysctl tuning. Idempotent on re-run.
#
# Root (or passwordless sudo) required.
#
# Intended for cluster operators reproducing MLPerf submissions on bare metal.
# For air-gapped nodes, mirror the APT/YUM repos referenced below and set
# MLPERF_OFFLINE_MIRROR in env.

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

CUDA_REPO_UBUNTU_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_REPO_UBUNTU_URL_2404="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
NVIDIA_CT_KEYRING_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
NVIDIA_CT_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
DOCKER_GPG_URL="https://download.docker.com/linux/ubuntu/gpg"
ENROOT_DEB_URL_TEMPLATE="https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot_3.5.0-1_%ARCH%.deb"
ENROOT_HOOKS_DEB_URL_TEMPLATE="https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot+caps_3.5.0-1_%ARCH%.deb"
PYXIS_VERSION="0.20.0"
# Immutable SHA corresponding to the v0.20.0 tag (at time of writing).
# Override via PYXIS_SHA=<sha> env var.
PYXIS_SHA="${PYXIS_SHA:-5fa3c38c73aab30adb9f7a1ff3c37b89d0938a43}"
# Slurm version here is informational. The script installs the distro's
# packaged Slurm (apt: slurm-wlm, dnf: slurm). Override by configuring an
# upstream Slurm repo before running this script if a specific version is
# required.
SLURM_VERSION_NOTE="distro-packaged (see comment above)"

[[ $EUID -eq 0 ]] || { command -v sudo >/dev/null || die "Need root or sudo"; SUDO=sudo; }
: "${SUDO:=}"

OS_ID=$(. /etc/os-release; echo "$ID")
OS_VER=$(. /etc/os-release; echo "$VERSION_ID")
ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
[[ "$ARCH" == "x86_64" ]] && ARCH_DEB=amd64 || ARCH_DEB="$ARCH"

case "$OS_ID" in
    ubuntu) PKG="apt-get" ;;
    rhel|centos|rocky|almalinux) PKG="dnf" ;;
    *) die "Unsupported OS: $OS_ID" ;;
esac
info "OS: $OS_ID $OS_VER   arch: $ARCH_DEB   pkg: $PKG"

pkg_update() {
    # Explicit if/else — the prior `A && B || C` form would silently run
    # `$PKG makecache` after `apt-get update` failures, which on APT is an
    # invalid subcommand.
    if [[ "$PKG" == "apt-get" ]]; then
        $SUDO apt-get update
    else
        $SUDO $PKG makecache -y
    fi
}
pkg_install() {
    if [[ "$PKG" == "apt-get" ]]; then
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "$@"
    else
        $SUDO $PKG install -y "$@"
    fi
}

# ----------------------------------------------------------
# Step 1 — base build chain
# ----------------------------------------------------------
say "Step 1: base packages"
pkg_update
if [[ "$PKG" == "apt-get" ]]; then
    pkg_install ca-certificates curl gnupg lsb-release build-essential dkms \
                "linux-headers-$(uname -r)" hwdata jq
else
    pkg_install curl gnupg2 tar gcc gcc-c++ make kernel-devel kernel-headers \
                hwdata jq redhat-lsb-core
fi

# ----------------------------------------------------------
# Step 2 — NVIDIA driver via CUDA repo (recommended)
# ----------------------------------------------------------
say "Step 2: NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    info "nvidia-smi present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    if [[ "$OS_ID" == "ubuntu" ]]; then
        case "$OS_VER" in
            22.04) URL="$CUDA_REPO_UBUNTU_URL"   ;;
            24.04) URL="$CUDA_REPO_UBUNTU_URL_2404" ;;
            *)     die "Unsupported Ubuntu version: $OS_VER" ;;
        esac
        TMPDEB=$(mktemp --suffix=.deb)
        curl -fsSL -o "$TMPDEB" "$URL"
        $SUDO dpkg -i "$TMPDEB"
        rm -f "$TMPDEB"
        pkg_update
        pkg_install cuda-drivers
    else
        $SUDO $PKG config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        pkg_install cuda-drivers
    fi
    info "Driver installed. Reboot recommended before first GPU use."
fi

# ----------------------------------------------------------
# Step 3 — NVIDIA Container Toolkit
# ----------------------------------------------------------
say "Step 3: nvidia-container-toolkit"
if [[ "$PKG" == "apt-get" ]]; then
    curl -fsSL "$NVIDIA_CT_KEYRING_URL" | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL "$NVIDIA_CT_LIST_URL" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    pkg_update
    pkg_install nvidia-container-toolkit
else
    $SUDO $PKG config-manager --add-repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
    pkg_install nvidia-container-toolkit
fi

# ----------------------------------------------------------
# Step 4 — Docker CE
# ----------------------------------------------------------
say "Step 4: Docker CE"
if command -v docker >/dev/null 2>&1; then
    info "docker already installed: $(docker --version)"
else
    if [[ "$PKG" == "apt-get" ]]; then
        $SUDO install -m 0755 -d /etc/apt/keyrings
        curl -fsSL "$DOCKER_GPG_URL" | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        $SUDO chmod a+r /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$ARCH_DEB signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
            $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
        pkg_update
        pkg_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    else
        # dnf-family: use the distro-matching Docker repo (fedora vs centos/rhel).
        case "$OS_ID" in
            fedora) DOCKER_REPO_URL="https://download.docker.com/linux/fedora/docker-ce.repo" ;;
            *)      DOCKER_REPO_URL="https://download.docker.com/linux/centos/docker-ce.repo" ;;
        esac
        $SUDO $PKG config-manager --add-repo "$DOCKER_REPO_URL"
        pkg_install docker-ce docker-ce-cli containerd.io
    fi
fi
$SUDO nvidia-ctk runtime configure --runtime=docker
$SUDO systemctl enable --now docker

# ----------------------------------------------------------
# Step 5 — Enroot
# ----------------------------------------------------------
say "Step 5: Enroot"
if command -v enroot >/dev/null 2>&1; then
    info "enroot present: $(enroot version)"
else
    # Global replacement: "/PAT/REP" = first, "//PAT/REP" = all occurrences.
    # The template embeds %ARCH% mid-string so we need the global form.
    E1=${ENROOT_DEB_URL_TEMPLATE//%ARCH%/$ARCH_DEB}
    E2=${ENROOT_HOOKS_DEB_URL_TEMPLATE//%ARCH%/$ARCH_DEB}
    TD=$(mktemp -d)
    curl -fsSL -o "$TD/enroot.deb"      "$E1"
    curl -fsSL -o "$TD/enroot-hooks.deb" "$E2"
    $SUDO dpkg -i "$TD/enroot.deb" "$TD/enroot-hooks.deb" || pkg_install -f
    rm -rf "$TD"
fi
# Enroot ships its own /etc/enroot/enroot.conf.d/*.conf drop-ins via the deb
# package; no further config is required. Placeholder kept deliberately short
# so later provisioners (e.g. fabric.sh) know enroot is expected to be
# configured via the distro package.

# ----------------------------------------------------------
# Step 6 — Munge + Slurm
# ----------------------------------------------------------
say "Step 6: Munge + Slurm (${SLURM_VERSION_NOTE})"
if [[ "$PKG" == "apt-get" ]]; then
    pkg_install slurm-wlm slurm-wlm-basic-plugins slurmd slurmctld munge libmunge-dev libmunge2
else
    pkg_install munge munge-devel slurm slurm-slurmd slurm-slurmctld
fi
if [[ ! -f /etc/munge/munge.key ]]; then
    $SUDO dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key status=none
    $SUDO chown munge:munge /etc/munge/munge.key
    $SUDO chmod 0400 /etc/munge/munge.key
    info "Generated /etc/munge/munge.key — copy this to every node!"
fi
$SUDO systemctl enable --now munge || true

# ----------------------------------------------------------
# Step 7 — Pyxis SPANK plugin
# ----------------------------------------------------------
say "Step 7: Pyxis v${PYXIS_VERSION}"
# Check every common Slurm plugin dir (deb, rpm, source-built).
PYXIS_INSTALLED=0
for p in /usr/lib/x86_64-linux-gnu/slurm /usr/lib64/slurm /usr/local/lib/slurm \
         /usr/lib/slurm /opt/slurm/lib/slurm; do
    [[ -f "$p/spank_pyxis.so" ]] && { PYXIS_INSTALLED=1; info "Pyxis found at $p"; break; }
done
if (( PYXIS_INSTALLED == 0 )); then
    pkg_install git
    TD=$(mktemp -d)
    git clone https://github.com/NVIDIA/pyxis.git "$TD/pyxis"
    ( cd "$TD/pyxis" && git checkout "$PYXIS_SHA" && make && $SUDO make install )
    rm -rf "$TD"
fi
# Register with Slurm
$SUDO mkdir -p /etc/slurm/plugstack.conf.d
cat <<'EOF' | $SUDO tee /etc/slurm/plugstack.conf.d/pyxis.conf >/dev/null
required /usr/local/share/pyxis/pyxis.conf
EOF

# ----------------------------------------------------------
# Step 8 — sysctl / limits / hugepages
# ----------------------------------------------------------
say "Step 8: kernel tuning"
cat <<'EOF' | $SUDO tee /etc/sysctl.d/99-mlperf.conf >/dev/null
vm.nr_hugepages = 1024
net.core.rmem_max = 2147483647
net.core.wmem_max = 2147483647
net.core.rmem_default = 212992
net.core.wmem_default = 212992
net.ipv4.tcp_rmem = 4096 87380 2147483647
net.ipv4.tcp_wmem = 4096 65536 2147483647
net.ipv4.tcp_mem = 786432 1048576 26777216
net.core.netdev_max_backlog = 250000
net.core.somaxconn = 65535
EOF
$SUDO sysctl -p /etc/sysctl.d/99-mlperf.conf || true

cat <<'EOF' | $SUDO tee /etc/security/limits.d/99-mlperf.conf >/dev/null
* soft memlock unlimited
* hard memlock unlimited
* soft nofile 1048576
* hard nofile 1048576
* soft stack 67108864
* hard stack 67108864
EOF

# ----------------------------------------------------------
# Step 9 — MIG helper script (drop-in)
# ----------------------------------------------------------
say "Step 9: installing /usr/local/sbin/mlperf-mig"
$SUDO tee /usr/local/sbin/mlperf-mig >/dev/null <<'EOS'
#!/usr/bin/env bash
# Configure MIG partitioning via nvidia-smi. Usage: mlperf-mig enable|disable|list|apply <profile>
set -e
case "${1:-}" in
    enable)  nvidia-smi -mig 1 ;;
    disable) nvidia-smi -mig 0 ;;
    list)    nvidia-smi mig -lgip ;;
    apply)   shift; nvidia-smi mig -cgi "$@" -C ;;
    *) echo "Usage: mlperf-mig {enable|disable|list|apply <profile>}"; exit 1 ;;
esac
EOS
$SUDO chmod 0755 /usr/local/sbin/mlperf-mig

say "Bootstrap complete."
info "Next steps:"
info "  1. Reboot (if NVIDIA driver was just installed)."
info "  2. On the head node: edit /etc/slurm/slurm.conf with NodeName / PartitionName."
info "  3. Copy /etc/munge/munge.key to every compute node."
info "  4. systemctl enable --now slurmd (compute) / slurmctld (head)."
info "  5. Verify: nvidia-smi, docker run --gpus all nvidia/cuda:12.4.0-base nvidia-smi,"
info "     sinfo, srun --pty bash"
