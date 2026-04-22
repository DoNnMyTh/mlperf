# Troubleshooting

Live matrix of real failures operators have hit and how to unblock.

## Environment

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ERROR: Bash >= 4 required` | macOS default `/bin/bash` is 3.2 | `brew install bash` and run with `/opt/homebrew/bin/bash mlperf.sh` |
| `ERROR: non-interactive stdin` | Running under `ssh … 'bash mlperf.sh'` or piped CI | Use a real TTY (`ssh -t`) or pass `MLPERF_AUTO_YES=1` with `MLPERF_CONFIG_FILE=…` |
| `Docker daemon unreachable` | Docker Desktop not started | Start Docker; script waits up to 3 min |
| Git Bash on Windows rewrites `/data` to `C:/Program Files/Git/data` | MSYS path translation | Handled — script sets `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"` on Windows. Don't unset. |

## Container / image

| Symptom | Cause | Fix |
|---------|-------|-----|
| `unauthorized: authentication required` during pull | Private registry or rate limit | Accept the `docker login <host>` prompt |
| `docker pull failed (exit N) — not an auth error` | Network, disk, typo'd tag | Retry; verify image name; check `docker system df` |
| `CUDA Error: no kernel image is available` | Image compiled without your GPU's sm arch | Pull the `-sm89` variant for Ada, or rebuild with `NVTE_CUDA_ARCHS="89;100a;103a"` |
| `FileNotFoundError: Expected /preproc_data/… to exist` | Dataset not mounted, or wrong layout | Confirm `DATADIR/<sub>/<marker>` exists; run `fix_nested_dataset` |

## Training

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OutOfMemoryError` in optimizer setup | GPU VRAM < ~80 GB | Use the custom smoke path; real 8B needs H100+ |
| `pipeline-model-parallel size should be greater than 1 with interleaved schedule` | `INTERLEAVED_PIPELINE != 0` with PP=1 | Custom smoke sets `INTERLEAVED_PIPELINE=0`; for real configs verify matching topology |
| `Can not use sequence parallelism without tensor parallelism` | TP=1 with `SEQ_PARALLEL=True` | Custom smoke sets `SEQ_PARALLEL=False`; for real configs verify topology |
| Training hangs at "init NCCL" | Fabric manager down (NVSwitch) or IB link down | `systemctl status nvidia-fabricmanager`; `ibstat`; run `tools/provision/fabric.sh` |
| Slurm job stuck in `CONFIGURING` | Node drain or Gres mismatch | `sinfo -R` shows reason; `scontrol update NodeName=… State=RESUME` |

## Compliance / submission

| Symptom | Cause | Fix |
|---------|-------|-----|
| `compliance_checker: ValueError: event … missing` | MLLOG truncated; run crashed before `run_stop` | Rerun; ensure `SIGTERM` is proxied to Python |
| `two runs with identical seed converged to different values` | Non-determinism: dataloader worker count, rng init | Pin `num_workers`, `torch.use_deterministic_algorithms(True)`, verify `SEED` is set before `DataLoader` |
| `systems/<desc>.json still contains FIXME` | Forgot to edit the stub from `tools/submit.sh` | Replace every FIXME; re-run `tools/compliance_attest.sh` |
| `gh pr create: validation failed: Head ref must be a branch` | Ran submit_pr.sh without pushing the branch first | The script auto-pushes; check git push output for auth / 2FA errors |

## Provisioning

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvidia-fabricmanager-<cuda>: no such package` on RHEL | Wrong repo or older CUDA layout | `fabric.sh` now tries `cuda-drivers-fabricmanager-<cuda>` and `nvidia-fabric-manager`. Set `CUDA_VERSION_SUFFIX=12-6` if needed. |
| Munge key mismatch across nodes | `/etc/munge/munge.key` not identical | Use the Ansible playbook's Phase 2 (slurp from head, copy to compute) |
| Pyxis `--container-image` ref form rejected | Pyxis version mismatch | Use the `docker://` form (option B in Step 2's Pyxis ref picker) |
| BeeGFS mount doesn't persist across reboot | On your distro `beegfs-client.service` is templated | `systemctl enable beegfs-client@<id>` after editing `/etc/beegfs/beegfs-mounts.conf` |

## Filing a new entry

If you hit something not listed here, open a GitHub issue on `DoNnMyTh/mlperf` with:
- The exact command you ran.
- The first 20 lines of the error.
- `bash --version`, `uname -a`, `docker info --format '{{.ServerVersion}}'`.
- Whether it reproduces with a fresh `git clone`.
