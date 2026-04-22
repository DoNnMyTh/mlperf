#!/usr/bin/env bash
# Shared-filesystem client mount for MLPerf compute nodes.
#
# Mounts one of {NFSv4, Lustre, BeeGFS, SMB/CIFS} read-write at the requested
# local path, installs the matching client package, adds an /etc/fstab entry
# so the mount persists, and re-mounts it. Server-side provisioning is out of
# scope — operators must point at an existing export.
#
# Intended to be run after bootstrap_node.sh on every compute node (via
# Ansible loop, ClusterSSH, or manual SSH).

set -u
set -o pipefail

say()  { printf "\n==> %s\n" "$*"; }
info() { printf "    %s\n" "$*"; }
err()  { printf "ERROR: %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }
ask()  { local p="$1" d="${2-}" v; if [[ -n "$d" ]]; then read -r -p "$p [$d]: " v; echo "${v:-$d}"; else read -r -p "$p: " v; echo "$v"; fi; }
ask_req(){ local p="$1" v; while :; do read -r -p "$p: " v; [[ -n "$v" ]] && { echo "$v"; return; }; err "required"; done; }
yesno(){ local p="$1" d="${2-y}" v; while :; do read -r -p "$p (y/n) [$d]: " v; v="${v:-$d}"
          case "$v" in [Yy]*) return 0;; [Nn]*) return 1;; esac; done; }
pick() { local p="$1"; shift; local i=1; for o in "$@"; do printf "  [%d] %s\n" "$i" "$o" >&2; i=$((i+1)); done
         local v; while :; do read -r -p "$p [1]: " v; v="${v:-1}"; [[ "$v" =~ ^[0-9]+$ ]] && (( v>=1 && v<=$# )) && { echo "$v"; return; }; done; }

[[ $EUID -eq 0 ]] || { command -v sudo >/dev/null || die "Need root or sudo"; SUDO=sudo; }
: "${SUDO:=}"

OS_ID=$(. /etc/os-release; echo "$ID")
case "$OS_ID" in
    ubuntu|debian) PKG=apt-get ;;
    rhel|centos|rocky|almalinux|fedora) PKG=dnf ;;
    *) die "Unsupported OS: $OS_ID" ;;
esac
pkg_install(){ if [[ "$PKG" == "apt-get" ]]; then DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "$@"; else $SUDO $PKG install -y "$@"; fi; }
pkg_update(){ if [[ "$PKG" == "apt-get" ]]; then $SUDO apt-get update; else $SUDO $PKG makecache -y; fi; }

FSTYPE_CHOICES=("nfs4  (NFS v4)" "lustre" "beegfs" "cifs  (SMB)")
sel=$(pick "Shared filesystem type" "${FSTYPE_CHOICES[@]}")
case "$sel" in 1) FSTYPE=nfs4 ;; 2) FSTYPE=lustre ;; 3) FSTYPE=beegfs ;; 4) FSTYPE=cifs ;; esac
info "Filesystem: $FSTYPE"

MNT="$(ask 'Local mount point' /mnt/shared)"
if [[ ! -d "$MNT" ]]; then $SUDO mkdir -p "$MNT"; fi

case "$FSTYPE" in
    nfs4)
        SERVER="$(ask_req 'NFS server (host or IP)')"
        EXPORT="$(ask_req 'Server export path (e.g. /srv/nfs/shared)')"
        OPTS="$(ask 'Mount options' 'nfsvers=4.1,hard,noatime,async,rsize=1048576,wsize=1048576,tcp,_netdev')"
        pkg_update; pkg_install nfs-common || pkg_install nfs-utils
        SOURCE="${SERVER}:${EXPORT}"
        FSTAB_LINE="$SOURCE $MNT nfs4 $OPTS 0 0"
        ;;
    lustre)
        pkg_update
        # Lustre client RPMs/DEBs are distributed by Whamcloud/Intel; end users
        # typically install via the vendor's repo. Here we assume the repo is
        # already configured and install the client package.
        if [[ "$PKG" == "apt-get" ]]; then
            pkg_install lustre-client-modules-"$(uname -r)" lustre-client-utils \
                || die "Lustre packages not available. Configure Whamcloud repo first."
        else
            pkg_install lustre-client lustre-client-dkms \
                || die "Lustre packages not available. Configure Whamcloud repo first."
        fi
        MGS="$(ask_req 'Lustre MGS:fsname (e.g. 10.0.0.1@o2ib:/lfs)')"
        OPTS="$(ask 'Mount options' 'flock,noatime,_netdev')"
        SOURCE="$MGS"
        FSTAB_LINE="$SOURCE $MNT lustre $OPTS 0 0"
        ;;
    beegfs)
        pkg_update
        if [[ "$PKG" == "apt-get" ]]; then
            # BeeGFS requires upstream repo; assume configured. If not, guide.
            if ! apt-cache show beegfs-client >/dev/null 2>&1; then
                die "beegfs-client not in apt cache. Add BeeGFS repo (https://doc.beegfs.io/latest/)."
            fi
            pkg_install beegfs-client beegfs-helperd beegfs-utils
        else
            if ! $SUDO $PKG list beegfs-client 2>&1 | grep -q beegfs; then
                die "beegfs-client not in dnf repos. Add BeeGFS repo first."
            fi
            pkg_install beegfs-client beegfs-helperd beegfs-utils
        fi
        MGMTD="$(ask_req 'BeeGFS management node (host:port)')"
        $SUDO /opt/beegfs/sbin/beegfs-setup-client -m "$MGMTD" -n beegfs
        # beegfs-helperd is always a service.
        $SUDO systemctl enable --now beegfs-helperd \
            || warn "beegfs-helperd not enabled — check 'systemctl status beegfs-helperd'."
        # beegfs-client unit *name* varies: newer distros ship 'beegfs-client.service'
        # driven by /etc/beegfs/beegfs-mounts.conf; older distros use '@'-templated
        # instances (beegfs-client@<id>.service). Try the canonical unit first.
        if systemctl list-unit-files | grep -qE '^beegfs-client\.service'; then
            $SUDO systemctl enable --now beegfs-client
        else
            warn "beegfs-client.service not found. Configure /etc/beegfs/beegfs-mounts.conf and start the correct unit manually."
        fi
        info "BeeGFS client configured. Mount appears per /etc/beegfs/beegfs-mounts.conf (default: /mnt/beegfs)."
        exit 0
        ;;
    cifs)
        SERVER="$(ask_req 'SMB server (//server/share)')"
        USERNAME="$(ask_req 'SMB username')"
        CRED_FILE="/etc/cifs-mlperf.cred"
        yesno "Store credentials in $CRED_FILE (0600)?" y || die "Aborted."
        read -rsp 'SMB password: ' PASS; echo
        # Write via a temp file (mktemp in /tmp, mode 600) then sudo-move it.
        # This avoids passing the password through the shell command line or
        # through an unquoted here-doc (where $ or backticks in the password
        # would be expanded/executed).
        TMP_CRED=$(mktemp)
        chmod 600 "$TMP_CRED"
        printf 'username=%s\npassword=%s\n' "$USERNAME" "$PASS" > "$TMP_CRED"
        $SUDO install -m 0600 -o root -g root "$TMP_CRED" "$CRED_FILE"
        shred -u "$TMP_CRED" 2>/dev/null || rm -f "$TMP_CRED"
        unset PASS
        pkg_update
        pkg_install cifs-utils
        OPTS="$(ask 'Mount options' 'credentials=/etc/cifs-mlperf.cred,uid=0,gid=0,vers=3.1.1,_netdev,nofail')"
        SOURCE="$SERVER"
        FSTAB_LINE="$SOURCE $MNT cifs $OPTS 0 0"
        ;;
esac

say "Registering /etc/fstab entry"
if grep -F -q " $MNT " /etc/fstab 2>/dev/null; then
    info "Existing fstab entry found for $MNT — replacing."
    $SUDO sed -i "\\| $MNT |d" /etc/fstab
fi
echo "$FSTAB_LINE" | $SUDO tee -a /etc/fstab >/dev/null
info "fstab: $FSTAB_LINE"

say "Mounting"
$SUDO mount "$MNT" || die "mount failed — check credentials/connectivity."
info "Mounted:"
df -h "$MNT"

say "Done. Re-run on every compute node."
info "Verification on another node after mount:"
info "  mount | grep $MNT"
info "  touch $MNT/.mlperf-write-check && rm $MNT/.mlperf-write-check"
