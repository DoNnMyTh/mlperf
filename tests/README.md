# tests/

Bats-core test suite for the pure-shell helpers in `lib/common.sh` and a
handful of routines in `mlperf.sh` that can be unit-tested without a GPU.

## Running

```bash
# Install bats-core (once):
git clone https://github.com/bats-core/bats-core.git ~/.bats
~/.bats/install.sh /usr/local    # or ~/.local

# Run the suite:
bats tests/
```

## What is covered

| Test file                       | Covers                                     |
|---------------------------------|--------------------------------------------|
| `test_common.bats`              | `validate_path`, `retry`, `free_gb` regex, `random_port` range, `notify` noop |
| `test_gpu_shim.bats`            | `detect_gpus` against the fake `nvidia-smi` shim in `tests/fixtures/` |
| `test_manifest_schema.bats`     | Every `workloads/*.manifest.sh` sources cleanly with `set -u` and defines the required `WL_*` fields |

## GPU shim

`tests/fixtures/bin/nvidia-smi` is a ~20-line bash script that mocks the
subset of `nvidia-smi` output the scripts consume (`--query-gpu=...
--format=csv,noheader`). Prepending `tests/fixtures/bin` to `$PATH` lets
`detect_gpus` think it is on a machine with four fake GPUs, making the GPU
path testable on a laptop with no CUDA install.
