# Architecture

## Component diagram

```mermaid
flowchart TB
    user([Operator])
    subgraph Repo[mlperf repo]
        driver[mlperf.sh<br/>driver]
        lib[lib/common.sh<br/>retry · state · notify · pins]
        workloads[workloads/*.manifest.sh<br/>WL_* contract]
        tools[tools/<br/>convert · eval · submit · compliance ·<br/>submit_pr · report · build_all · SBOM]
        provision[tools/provision/<br/>bootstrap_node · fabric · shared_fs ·<br/>ansible/]
        tests[tests/<br/>bats + nvidia-smi shim]
    end
    subgraph External
        mlcommons[(mlcommons/<br/>training_results_vX.Y)]
        hub[(Docker Hub<br/>donnmyth/mlperf-nvidia)]
        slurm[(Slurm cluster<br/>Pyxis + Enroot)]
        mlpolicies[(mlcommons/<br/>training_policies)]
    end

    user -->|bash mlperf.sh| driver
    driver -->|source| lib
    driver -->|pick + source| workloads
    driver -->|dispatch| slurm
    driver -->|docker run| hub
    driver -->|clone| mlcommons
    tools -->|mlperf_logging 3.1.0| hub
    tools -->|gh CLI| mlcommons
    provision -->|APT/DNF| Distro[(Distro repos)]
    provision -->|Pyxis SHA| Pyxis[(github.com/NVIDIA/pyxis)]
    tests -->|bats-core| lib
    tests -->|bats-core| workloads
```

## Phase state machine

```mermaid
stateDiagram-v2
    [*] --> Preflight: start
    Preflight --> WorkloadPick: git ok
    WorkloadPick --> Repo: manifest loaded
    Repo --> Container
    Container --> Dataset
    Dataset --> Config
    Config --> Runtime
    Runtime --> Launcher
    Launcher --> sbatch: MLPerf compliant
    Launcher --> srun
    Launcher --> docker: single-node
    Launcher --> bare
    Launcher --> prepare: staging only
    sbatch --> Compliance: tools/compliance.sh
    docker --> [*]: non-compliant warning
    bare --> [*]: non-compliant warning
    prepare --> [*]: exit 0 with commands
    Compliance --> Attest: tools/compliance_attest.sh
    Attest --> Submit: tools/submit.sh
    Submit --> PR: tools/submit_pr.sh
    PR --> [*]
```

## Key design invariants

1. **Manifest-driven** — the driver has no workload-specific code. Adding a workload means adding a manifest.
2. **Interactive by default, scripted by exception** — `MLPERF_AUTO_YES=1` or `MLPERF_CONFIG_FILE=...` flips every prompt to its default without changing the interactive behaviour.
3. **No GitHub Actions** — every check runs locally or via a user-chosen CI (Jenkins, buildkite). `.git-hooks/pre-commit` for gitleaks; `tests/` for bats; `tools/sbom_and_sign.sh` for cosign.
4. **Pinned upstream** — `lib/common.sh` exports `PIN_*` constants (mlperf_logging version, Pyxis SHA, Enroot version). Changing a pin is a single-file commit.
5. **MLPerf compliance is gated, not promised** — any launcher other than `sbatch run.sub` prints a "not MLPerf-compliant" notice; `tools/compliance.sh` and `tools/compliance_attest.sh` mechanically validate submission artefacts.
