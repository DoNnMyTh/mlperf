# Multi-Node Cluster Orchestration (Ansible)

Wraps [`bootstrap_node.sh`](../bootstrap_node.sh) into an Ansible role so a fresh cluster can be stood up with one playbook run.

## Layout

```
ansible/
├── site.yml                                  # top-level playbook (4 plays)
├── inventory.example.ini                     # copy to inventory.ini, edit
├── group_vars/
│   └── all.example.yml                       # copy to all.yml, edit
└── roles/
    ├── mlperf_node/                          # runs bootstrap_node.sh on every host
    │   └── tasks/main.yml
    └── slurm_head/                           # renders slurm.conf + starts slurmctld
        ├── tasks/main.yml
        ├── handlers/main.yml
        └── templates/slurm.conf.j2
```

## Prerequisites

- Ansible ≥ 2.14 on the control host.
- SSH reachability + `sudo` on every managed node (use `--ask-become-pass` or pre-configure passwordless sudo).
- Ubuntu 22.04/24.04 or RHEL 9 on managed nodes.
- Network connectivity between nodes and to Docker Hub / GitHub / NVIDIA repos during provisioning.

## Usage

```bash
cd tools/provision/ansible

# 1. Inventory
cp inventory.example.ini inventory.ini
$EDITOR inventory.ini                # set head + compute hosts

# 2. Variables
cp group_vars/all.example.yml group_vars/all.yml
$EDITOR group_vars/all.yml           # accelerator_type, gpus_per_node, etc.

# 3. Dry-run
ansible-playbook -i inventory.ini site.yml --check --diff

# 4. Apply
ansible-playbook -i inventory.ini site.yml
```

## Play structure

The `site.yml` runs four plays, in order:

1. **Phase 1 — mlperf_node on every host.** Stages and executes `bootstrap_node.sh`. Installs NVIDIA driver, container toolkit, Docker, Enroot, Munge, Slurm, Pyxis, kernel tuning, MIG helper.
2. **Phase 2 — distribute munge.key from head to compute.** Copies `/etc/munge/munge.key` generated on the head node to every compute node and restarts `munge`.
3. **Phase 3 — slurm_head on the head node.** Renders `/etc/slurm/slurm.conf` from an inventory-derived template (one `NodeName` line per compute host with the correct `Gres=gpu:<type>:<count>`), installs the Pyxis plugstack entry, and starts `slurmctld`.
4. **Phase 4 — start slurmd on compute nodes.** Ensures `slurmd` is enabled and running.

## Idempotency

All tasks are idempotent. Re-running `site.yml` on a fully provisioned cluster is a no-op beyond validating current state. `bootstrap_node.sh` itself short-circuits when the corresponding component is already installed.

## Common overrides

```yaml
# group_vars/all.yml
accelerator_type: b200
gpus_per_node: 8
partition_max_time: "48:00:00"
```

## Running from the repo checkout

The role expects `bootstrap_node.sh` to be available at `tools/provision/bootstrap_node.sh` relative to the playbook. When executing from the checked-out repo this is satisfied. If you publish the role elsewhere, adjust the `ansible.builtin.copy` source path in `roles/mlperf_node/tasks/main.yml`.

## Not covered by this playbook

- NVSwitch / fabric manager configuration — see [`../fabric.sh`](../fabric.sh) *(to be added)*.
- InfiniBand / RoCE driver tuning — see [`../fabric.sh`](../fabric.sh) *(to be added)*.
- Shared filesystem mount (Lustre / BeeGFS / NFS) — see [`../shared_fs.sh`](../shared_fs.sh) *(to be added)*.
- User accounts, quotas, PAM — site-specific.
