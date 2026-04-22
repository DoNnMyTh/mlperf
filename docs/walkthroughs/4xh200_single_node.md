# Walkthrough — 4×H200 NVLink single-node cluster

Covers: one node, four H200 GPUs wired via NVLink, no Slurm required.
Goal: a reproducible smoke + custom training run.

## Hardware / Software

| Item | Expected |
|------|----------|
| GPU  | 4× H200 (sm_90, 141 GB each), NVLink or NVSwitch |
| Host OS | Ubuntu 22.04 or RHEL 9 |
| Drivers | NVIDIA ≥ 560 ; nvidia-container-toolkit ; docker ≥ 24 |
| Disk | ≥ 200 GB free on the dataset drive |
| Network | Outbound to Docker Hub, GitHub, `training.mlcommons-storage.org` |

## Gotcha #1 — no published image covers sm_90

The two Docker Hub tags ship these kernel archs:

| Tag | `NVTE_CUDA_ARCHS` | H200 works? |
|-----|-------------------|:-----------:|
| `…-blackwell` | `100a;103a`      | ✗ |
| `…-sm89`      | `89;100a;103a`   | ✗ |

You must **build locally** with the Dockerfile patch that adds sm_90:

```
Step 2: container source
  → 1  (docker: build locally)
  → Patch Dockerfile NVTE_CUDA_ARCHS to add 89 (RTX 40xx/Ada)? y
```

The manifest's `WL_DOCKERFILE_PATCH_TO` now expands to `89;90;100a;103a`, so a single "yes" covers both Ada and Hopper. Build time ≈ 25 min (TE + Apex wheel compile).

## Gotcha #2 — no 1-node × 4-GPU *canned* config is published

**The driver already detects all 4 GPUs** and Step 5 (`GPUs to use (1..N)`) lets you pick 1–4. The concern is only about the topology declared inside the selected `config_*.sh`: its `TENSOR_MODEL_PARALLEL × PIPELINE_MODEL_PARALLEL × CONTEXT_PARALLEL` product (model parallelism = `mp`) must be ≤ your world size (4), otherwise NCCL init crashes with a world-size mismatch.

The driver now detects this after sourcing the config and offers auto-adapt:

```
WARN: Config needs TP*PP*CP=8 GPUs, but you picked 4.
WARN: World size < model parallelism would fail at NCCL init.
Auto-adapt parallelism to fit 4 GPUs? (y/n) [y]: y
    Adapted: TP=2 PP=1 CP=2 (mp=4, dp=1)
```

It halves `CP` first, then `PP`, then `TP` (always by factors of 2) until `mp ≤ NGPU`. It also resets `INTERLEAVED_PIPELINE=0` when `PP=1` and `SEQ_PARALLEL=False` when `TP=1` so the run doesn't trip those constraints.

The smallest published configs are:

| Workload | Smallest config | Topology |
|----------|-----------------|----------|
| llama31_8b | `config_GB200_2x4x2xtp1pp1cp1_8b_fp4.sh` | 2×4 = 8 GPUs, FP4 only |
| llama31_405b | `config_GB200_128x4x…` | 512+ GPUs |
| llama2_70b_lora | `config_GB200_2x4x2xtp1pp1cp2_fp4.sh` | 2×4 = 8 GPUs, FP4 |

On a single 4-GPU H200 node, two viable paths:

### Path A — custom smoke (any workload with `WL_SMOKE_SUPPORTED=1`)

Runs a 2-layer synthetic-data training loop on one GPU to validate plumbing. Not MLPerf-compliant but proves every component works on your stack.

```
Step 4: config → last option (custom single-GPU smoke)
Step 5: GPUs  → 1 / 4
Step 6: launch → docker smoke
```

### Path B — adapted 2×4 config reduced to 1×4

Copy the published `config_GB200_2x4x2xtp1pp1cp1_8b_fp4.sh`, edit:

```bash
export DGXNNODES=1
export DGXNGPU=4
export MINIBS=1
# Halve parallelism: tp × pp × cp must divide world size.
export TENSOR_MODEL_PARALLEL=1
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=2        # world=4 / mp=2 => dp=2
# H200 supports FP8 but not FP4 via Transformer Engine 2.8. Drop FP4:
export FP4=False
export FP8=True
export FP8_HYBRID=True
```

Save as `config_H200_1x4x1xtp1pp1cp2_8b.sh` inside the workload impl dir and pick it in Step 4.

This produces a valid training run on your hardware but is **not a closed-division MLPerf submission** — you would need to upstream the config and reproduce the reference quality target across N successful seeds to submit.

## End-to-end dry-run (one-shot, non-interactive)

```bash
# Fully scripted — no prompts. See docs/examples/4xh200.env for the template.
export MLPERF_CONFIG_FILE=$PWD/docs/examples/4xh200.env
bash mlperf.sh
```

## Expected resource usage (llama31_8b, Path A)

| Step | Wall time | Disk | VRAM (per GPU) |
|------|-----------|------|-----------------|
| Clone repo | 30 s | 2 GB | – |
| Build image | 25 min | 38 GB | – |
| Download dataset | 6–8 h | 80 GB | – |
| Custom smoke (3 steps) | 1 min | <1 GB | 4 GB |
| Real 1×4 run (edited cfg) | ~6 h/epoch | +checkpoints | 60–90 GB |

## Validation checklist

After a run:

```bash
bash tools/compliance.sh            # mechanical gate
bash tools/eval.sh                  # STATUS=llama31_8b:PASS
bash tools/report.py --logs /mnt/logs --out /mnt/logs/report.html
```

Expect `compliance_checker` to PASS and `quality target : MET` once the run converges to log_perplexity ≤ 3.3.

## If it fails

Consult [`../TROUBLESHOOTING.md`](../TROUBLESHOOTING.md) — the H200 section covers the two most common failure modes (`CUDA Error: no kernel image` and `FP4 not supported`).
