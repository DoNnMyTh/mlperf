#!/usr/bin/env bats
# Every workloads/*.manifest.sh must source cleanly and define the required
# WL_* fields. Acts as a schema check — nothing in the repo can add or
# rename a field without updating the tests.

REQUIRED_FIELDS=(
    WL_NAME
    WL_DISPLAY
    WL_IMPL_SUBDIR
    WL_IMAGE_TAG_BASE
    WL_DATASET_SUBDIR
    WL_PREPROC_HOST_SUBPATH
    WL_PREPROC_MOUNT
    WL_CONFIG_GLOB
    WL_ENTRY
    WL_CONTAINER_WORKDIR
    WL_DOC_URL
)

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/.." && pwd)"
}

@test "each manifest sources cleanly with set -u" {
    for m in "$REPO_ROOT/workloads/"*.manifest.sh; do
        run bash -c "set -u; source '$m'"
        [[ "$status" -eq 0 ]] || { echo "FAIL sourcing $m: $output"; return 1; }
    done
}

@test "each manifest defines required WL_* fields" {
    for m in "$REPO_ROOT/workloads/"*.manifest.sh; do
        (
            set -u; source "$m"
            for f in "${REQUIRED_FIELDS[@]}"; do
                v="${!f-}"
                [[ -n "$v" ]] || { echo "$m: $f missing or empty"; exit 1; }
            done
        )
    done
}

@test "no manifest forgets to declare smoke-env as an array" {
    for m in "$REPO_ROOT/workloads/"*.manifest.sh; do
        grep -q '^declare -a WL_SMOKE_ENV' "$m" \
            || grep -q '^WL_SMOKE_ENV=(' "$m" \
            || { echo "$m: WL_SMOKE_ENV must be an array"; return 1; }
    done
}
