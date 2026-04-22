# Walkthrough — Llama 3.1 8B (NeMo)

End-to-end path from a fresh clone to a submittable result. Expected wall
time annotated per step.

## Preconditions

| Item | Check |
|------|-------|
| Host | Ubuntu 22.04 + Docker Desktop (≥24.0), or Linux with Docker + nvidia-container-toolkit |
| GPU  | ≥ 80 GB VRAM (H100 / H200 / B200) for compliance; any Ada (4080/4090) for smoke only |
| Disk | ≥ 200 GB free on dataset drive; ≥ 80 GB on build drive |
| Network | Outbound to Docker Hub, GitHub, MLCommons R2 |

## 1. Clone + initial config (~1 min)

```bash
git clone https://github.com/DoNnMyTh/mlperf.git
cd mlperf
bash mlperf.sh
# Step 0: preflight
# Step 0: pick workload → 5 (llama31_8b)
```

## 2. Pull prebuilt image (~5–10 min, cold; instant on warm cache)

```
Step 2: container source
  → 3  (docker: pull donnmyth/mlperf-nvidia:llama31_8b-pyt-sm89)
  → B  (docker:// pyxis ref)
```

**Why sm89 variant?** Ships Transformer Engine + Apex kernels compiled for sm_89 (Ada) alongside sm_100/103 (Blackwell) so smoke tests run on consumer GPUs.

## 3. Dataset download (~6–8 h wall, ~80 GB network)

```
Step 3: dataset
  → DATADIR: /mnt/data
  → Run download_8b.sh? y
```

What happens inside:
- `data_scripts/download_8b.sh` fetches the C4-preprocessed shards + Llama 3.1 tokenizer from `training.mlcommons-storage.org`.
- `data_scripts/cleanup_8b.sh` flattens the layout to `DATADIR/8b/{c4-*.bin,c4-*.idx,tokenizer/,LICENSE.txt,NOTICE.txt}`.
- `fix_nested_dataset` re-flattens if an earlier duplicate run created `DATADIR/8b/8b/…`.

## 4. Config pick + runtime (~30 s)

```
Step 4: config
  → config_GB200_18x4x1xtp1pp1cp2_8b.sh   # example for a GB200 system
Step 5: runtime
  → LOGDIR: /mnt/logs
  → SEED:   42
  → GPUs:   8 / 8 (full node)
```

## 5. Launch via sbatch (~30 min–2 h per run on GB200; 3 runs required for compliance)

```
Step 6: launch method → sbatch run.sub
  → --account  : my-account
  → --partition: gpu
  → NEXP       : 1
```

Repeat 3 times with different seeds (42, 43, 44).

## 6. Compliance + submission (~10 min)

```bash
bash tools/compliance.sh         # → /mnt/logs  (mechanical checks)
bash tools/submit.sh             # → build submission tarball
bash tools/compliance_attest.sh  # → reviewer-parity checks
bash tools/submit_pr.sh          # → open PR against mlcommons/training_results_v5.1
bash tools/report.py --logs /mnt/logs --out /mnt/logs/report.html
```

## Troubleshooting pointers

| Symptom | Pointer |
|---------|---------|
| `OOM in optimizer alloc` | You are on a 16 GB GPU. Expected for full 8B. Use custom smoke (Step 4 last option) to smoke the pipeline. |
| `CUDA Error: no kernel image` | Image was built without your GPU's sm arch. Switch to the `-sm89` tag or rebuild with `NVTE_CUDA_ARCHS="89;100a;103a"`. |
| `/results/container-env-*.log: No such file or directory` | `LOGDIR` mount does not exist or is not writable by the container user. `chmod 0777 $LOGDIR` or adjust uid mapping. |
| Slurm job dies at "GRES inconsistent" | `slurm.conf` `Gres=gpu:<type>:<count>` does not match the node's real hardware. Fix and `scontrol reconfigure`. |
