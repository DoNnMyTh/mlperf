# donnmyth/mlperf-nvidia

Reproducible MLPerf Training v5.1 NVIDIA-submission container images — pre-patched for Ada (sm_89), Hopper (sm_90), and Blackwell (sm_100/103).

Built by the [DoNnMyTh/mlperf](https://github.com/DoNnMyTh/mlperf) toolkit: a workload-agnostic, manifest-driven driver + operator tools for running every MLPerf Training v5.1 workload end-to-end (Docker, Enroot/Pyxis, or bare-metal).

## Short description

MLPerf Training v5.1 reference images for RTX 40xx / H100 / H200 / B200 / GB200 — patched TransformerEngine arch list.

## Available tags

| Workload | Tag | Arch coverage | Status |
|----------|-----|---------------|:------:|
| `llama31_8b` | `llama31_8b-pyt-blackwell` | `sm_100;sm_103` (B200/GB200/GB300) | live |
| `llama31_8b` | `llama31_8b-pyt-sm89` | + `sm_89` (RTX 40xx Ada) | live |
| `llama31_8b` | `llama31_8b-pyt-sm90` | + `sm_90` (H100/H200 Hopper) | live |
| `llama31_405b` | `llama31_405b-pyt` | upstream default (sm_100/103) | live |
| `llama31_405b` | `llama31_405b-pyt-sm90` | + `sm_90` | building |
| `llama2_70b_lora` | `llama2_70b_lora-pyt` | upstream default | live |
| `llama2_70b_lora` | `llama2_70b_lora-pyt-sm90` | + `sm_90` | building |
| `flux1` | `flux1-pyt` | upstream default | live |
| `flux1` | `flux1-pyt-sm90` | + `sm_90` | building |
| `retinanet` | `single_stage_detector-pyt` | upstream default | live |
| `rgat` | `graph_neural_network-dgl` | upstream default | live |
| `dlrm_dcnv2` | `recommendation-hugectr` | upstream default + mpi4py build fix | live |

`-sm90` variants are rebuilt with `NVTE_CUDA_ARCHS="89;90;100a;103a"` so kernels cover Ada + Hopper + Blackwell in one image.

## Quick pull

```bash
# RTX 40xx / L4 / L40 (Ada, sm_89)
docker pull donnmyth/mlperf-nvidia:llama31_8b-pyt-sm89

# H100 / H200 (Hopper, sm_90)
docker pull donnmyth/mlperf-nvidia:llama31_8b-pyt-sm90

# B200 / GB200 / GB300 (Blackwell, default)
docker pull donnmyth/mlperf-nvidia:llama31_8b-pyt-blackwell
```

## Running a benchmark

```bash
git clone https://github.com/DoNnMyTh/mlperf && cd mlperf
bash mlperf.sh
# Step 2 → docker: pull donnmyth/mlperf-nvidia:<tag>
```

See the [project README](https://github.com/DoNnMyTh/mlperf#readme) for full walkthroughs, 4×H200 single-node guide, and compliance workflow.

## Support

Issues, PRs, submission automation: https://github.com/DoNnMyTh/mlperf
