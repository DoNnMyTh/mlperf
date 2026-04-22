# Security Policy

## Supported Versions

This repository ships a single script, versioned via git. The tip of `master` is the only supported version.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security reports.**

Please use one of the following private channels:

1. **GitHub Private Vulnerability Reporting** — preferred.
   Open a report at: https://github.com/DoNnMyTh/mlperf/security/advisories/new
2. **Email** — `amit2cha@gmail.com` with subject line prefixed `[mlperf-security]`.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (exact inputs, OS, shell version).
- Any proof-of-concept code, limited to the minimum required to demonstrate the issue.
- Whether you have disclosed, or plan to disclose, the issue to third parties.

## Response Targets

| Stage | Target |
|-------|--------|
| Acknowledgement of report | 3 business days |
| Initial triage and severity assessment | 7 business days |
| Fix or mitigation for High/Critical | 30 days |
| Fix or mitigation for Low/Medium | Best effort |

## Scope

**In scope** (issues we will treat as security bugs):

- Command injection via user-supplied prompts, config filenames, or environment variables.
- Path traversal via `DATADIR`, `LOGDIR`, `REPO_DIR`, or `SQSH` paths that bypass `validate_path`.
- Credential leakage (e.g., writing secrets to stdout, logs, or `/tmp`).
- Unsafe automatic privilege escalation (`sudo` without confirmation).
- Container escape enabled by mount or `--gpus` configuration defaults.

**Out of scope** (report via normal GitHub issues if applicable):

- Bugs in upstream components (NeMo, Megatron-LM, Transformer Engine, Docker, Slurm). Report to the respective upstream projects.
- Denial of service via extreme user input (e.g., 10 GB `DATADIR` string).
- Issues only reproducible on forked or modified versions of the script.
- Social engineering of repository maintainers.

## Safe Harbor

Good-faith security research is welcomed. We will not pursue legal action against researchers who:

- Act in good faith and avoid privacy violations, service disruption, or destruction of data.
- Report findings privately through the channels above and allow a reasonable time for remediation before public disclosure.
- Limit testing to systems you own or have explicit permission to test.

## Hall of Fame

Researchers who make meaningful contributions to the security of this project are credited here (with their consent) after a fix ships.

_No entries yet._
