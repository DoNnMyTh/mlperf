#!/usr/bin/env bash
# Multi-architecture build + push (docker buildx) for one workload image.
#
# Produces a multi-arch manifest list at <hub>:<tag> containing linux/amd64
# and linux/arm64 images. Uses registry-backed BuildKit cache so subsequent
# builds reuse layers across machines.
#
# Prereq: `docker buildx create --use` once on the build machine, and
# QEMU emulators installed (`docker run --privileged --rm tonistiigi/binfmt
# --install all`) if cross-building.
#
# Usage:
#   tools/build_multiarch.sh --dir <impl-path> --tag <hub/image:tag> \
#                            [--platforms linux/amd64,linux/arm64] \
#                            [--cache-ref <hub/image:cache>]

set -u
set -o pipefail

DIR=""; TAG=""; PLATFORMS="linux/amd64,linux/arm64"; CACHE=""
while (( $# )); do
    case "$1" in
        --dir)        shift; DIR="$1" ;;
        --tag)        shift; TAG="$1" ;;
        --platforms)  shift; PLATFORMS="$1" ;;
        --cache-ref)  shift; CACHE="$1" ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -d "$DIR" ]]  || { echo "--dir required and must be a directory" >&2; exit 1; }
[[ -n "$TAG" ]]  || { echo "--tag required" >&2; exit 1; }
[[ -z "$CACHE" ]] && CACHE="${TAG%:*}:buildcache"

docker buildx inspect --bootstrap >/dev/null 2>&1 \
    || { echo "buildx not available; run 'docker buildx create --use' first" >&2; exit 1; }

echo "Platforms   : $PLATFORMS"
echo "Image tag   : $TAG"
echo "Cache ref   : $CACHE (registry)"
echo "Context dir : $DIR"

docker buildx build \
    --platform "$PLATFORMS" \
    --tag      "$TAG" \
    --cache-from "type=registry,ref=$CACHE" \
    --cache-to   "type=registry,ref=$CACHE,mode=max" \
    --push \
    "$DIR"
