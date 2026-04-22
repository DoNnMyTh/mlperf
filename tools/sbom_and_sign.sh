#!/usr/bin/env bash
# Generate an SBOM and sign a Docker Hub image with cosign.
#
# Produces:
#   <workdir>/<image-tag-safe>.sbom.spdx.json  (CycloneDX + SPDX)
#   <workdir>/<image-tag-safe>.sig              (cosign signature)
#   <workdir>/<image-tag-safe>.att.json         (cosign attestation over SBOM)
#
# Prereqs: docker, syft OR docker-sbom plugin, cosign >=2.2. Cosign can use
# keyless OIDC or a --key referenced from env.
#
# Usage:
#   COSIGN_EXPERIMENTAL=1 bash tools/sbom_and_sign.sh --image donnmyth/mlperf-nvidia:llama31_8b-pyt-sm89

set -u
set -o pipefail

IMAGE=""
WORKDIR="./artifacts"
while (( $# )); do
    case "$1" in
        --image)    shift; IMAGE="$1" ;;
        --workdir)  shift; WORKDIR="$1" ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -n "$IMAGE" ]] || { echo "--image required" >&2; exit 1; }

mkdir -p "$WORKDIR"
safe="${IMAGE//\//_}"; safe="${safe//:/_}"
sbom="$WORKDIR/$safe.sbom.spdx.json"
sig="$WORKDIR/$safe.sig"
att="$WORKDIR/$safe.att.json"

echo "[1/3] Generating SBOM -> $sbom"
if command -v syft >/dev/null 2>&1; then
    syft "$IMAGE" -o spdx-json > "$sbom"
elif docker sbom --help >/dev/null 2>&1; then
    docker sbom --format spdx-json "$IMAGE" > "$sbom"
else
    echo "ERROR: install 'syft' (https://github.com/anchore/syft) or enable Docker's 'sbom' plugin." >&2
    exit 1
fi

echo "[2/3] Signing image with cosign -> $sig"
if ! command -v cosign >/dev/null 2>&1; then
    echo "ERROR: install 'cosign' >= 2.2 (https://docs.sigstore.dev/cosign/system_config/installation/)" >&2
    exit 1
fi
# Keyless (OIDC) by default; override with COSIGN_KEY env var to use a key.
if [[ -n "${COSIGN_KEY:-}" ]]; then
    cosign sign --key "$COSIGN_KEY" --yes --output-signature "$sig" "$IMAGE"
else
    cosign sign --yes --output-signature "$sig" "$IMAGE"
fi

echo "[3/3] Attaching SBOM attestation (SLSA in-toto) -> $att"
if [[ -n "${COSIGN_KEY:-}" ]]; then
    cosign attest --key "$COSIGN_KEY" --yes --predicate "$sbom" \
        --type spdxjson --output-file "$att" "$IMAGE"
else
    cosign attest --yes --predicate "$sbom" \
        --type spdxjson --output-file "$att" "$IMAGE"
fi

echo "Done."
echo "  SBOM        : $sbom"
echo "  Signature   : $sig"
echo "  Attestation : $att"
echo "Verify later with: cosign verify-attestation <image> --type spdxjson"
