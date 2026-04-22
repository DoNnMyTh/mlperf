# Cluster Provisioning

Scripts for bringing a fresh cluster node into a state that can run MLPerf training — containerised, Slurm-scheduled, GPU-enabled.

## Contents

| File | Purpose |
|------|---------|
| `bootstrap_node.sh` | One-shot Bash installer for a single node. Idempotent. |
| `slurm.conf.example` | Template `slurm.conf` with GPU-aware `NodeName` / `PartitionName`. |

## What `bootstrap_node.sh` does

Runs on Ubuntu 22.04 / 24.04 or RHEL 9 (and compatible rebuilds). With root (or passwordless sudo):

1. Installs base build chain (`linux-headers`, `dkms`, etc.).
2. Installs the NVIDIA data-centre driver via the official CUDA repository.
3. Installs the NVIDIA Container Toolkit and configures the Docker runtime.
4. Installs Docker CE.
5. Installs Enroot 3.5.0 (DEB only; RHEL users build from source — see upstream).
6. Installs Munge and Slurm 23.11.6 (workload manager).
7. Builds and installs Pyxis 0.20.0 SPANK plugin; registers it with Slurm.
8. Applies MLPerf-friendly kernel tuning (`vm.nr_hugepages`, TCP buffers, limits).
9. Drops a `mlperf-mig` helper at `/usr/local/sbin/` for MIG partitioning.

On success it prints the post-install checklist: reboot, edit `slurm.conf`, distribute `munge.key`, start `slurmd` / `slurmctld`.

## Usage

```bash
# On each node (head + compute):
curl -fsSL https://raw.githubusercontent.com/DoNnMyTh/mlperf/master/tools/provision/bootstrap_node.sh | sudo bash
# ... reboot ...
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## MIG Partitioning

The `mlperf-mig` helper wraps `nvidia-smi mig`. MIG is required when sharing a single H100/H200/B200 between multiple Slurm jobs.

```bash
sudo mlperf-mig enable                                # set MIG mode on
sudo mlperf-mig list                                  # show available profiles
sudo mlperf-mig apply 9,9,9,9,9,9,9                   # seven 1g.10gb slices
sudo mlperf-mig disable                               # revert to full GPU
```

Reference profiles: see NVIDIA MIG User Guide (https://docs.nvidia.com/datacenter/tesla/mig-user-guide/).

## Slurm Cluster Head vs. Compute

### Head node
1. Generate `/etc/munge/munge.key` (bootstrap does this; the script prints a reminder to distribute).
2. Edit `/etc/slurm/slurm.conf` from the template (`ControlMachine`, `NodeName`, `Gres`, `Partition`).
3. `systemctl enable --now munge slurmctld`.

### Compute nodes
1. Copy `/etc/munge/munge.key` from the head (secure transport).
2. Copy identical `/etc/slurm/slurm.conf`.
3. `systemctl enable --now munge slurmd`.

## Verification

```bash
sinfo                              # should list nodes in idle/alloc
srun --pty bash -c 'nvidia-smi'    # run a GPU job via Slurm
srun --container-image=docker://donnmyth/mlperf-nvidia:llama31_8b-pyt-blackwell --pty nvidia-smi
```

## Not Handled Here

- Fabric manager / NVSwitch setup for GB200 — follow NVIDIA's vendor docs.
- InfiniBand / RoCE driver tuning — vendor-specific (Mellanox OFED).
- Lustre / GPFS / BeeGFS mount — site-specific; mount `/mnt/shared` before Slurm starts.
- User account provisioning and quota management.
- Log aggregation (Loki / Fluentd) — orthogonal to MLPerf.

For multi-node orchestration at scale, the recommended next step is to wrap this script in Ansible:

```yaml
# site.yml
- hosts: cluster
  become: true
  roles:
    - role: mlperf_node
```

See the commented template at the bottom of `bootstrap_node.sh` for the recommended Ansible role layout.
