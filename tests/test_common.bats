#!/usr/bin/env bats
# Unit tests for lib/common.sh helpers.

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/.." && pwd)"
    # shellcheck source=../lib/common.sh
    source "$REPO_ROOT/lib/common.sh"
    # ensure non-interactive tests don't pull from env
    unset MLPERF_AUTO_YES MLPERF_CONFIG_FILE MLPERF_NOTIFY_URL
    MLPERF_AUTO_YES=0
}

@test "validate_path rejects spaces" {
    run validate_path "/tmp/with space" t
    [[ "$status" -ne 0 ]]
    [[ "$output" =~ "must not contain spaces" ]]
}

@test "validate_path rejects semicolon" {
    run validate_path "/tmp/x;y" t
    [[ "$status" -ne 0 ]]
    [[ "$output" =~ "shell-special chars" ]]
}

@test "validate_path accepts plain paths" {
    run validate_path "/tmp/normal/path.txt" t
    [[ "$status" -eq 0 ]]
}

@test "retry returns 0 on first success" {
    run retry true
    [[ "$status" -eq 0 ]]
}

@test "retry gives up after N failures" {
    export MLPERF_RETRY_TRIES=2 MLPERF_RETRY_DELAY=0
    run retry false
    [[ "$status" -ne 0 ]]
}

@test "random_port returns a number in 20000-39999" {
    p=$(random_port)
    [[ "$p" =~ ^[0-9]+$ ]]
    (( p >= 20000 && p < 40000 ))
}

@test "notify is a no-op when MLPERF_NOTIFY_URL is unset" {
    run notify "hello"
    [[ "$status" -eq 0 ]]
    [[ -z "$output" ]]
}

@test "pinned versions are present" {
    [[ -n "$PIN_MLPERF_LOGGING" ]]
    [[ -n "$PIN_PYXIS_SHA" ]]
    [[ -n "$PIN_ENROOT_VERSION" ]]
}
