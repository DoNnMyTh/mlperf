# Contributing

Thank you for your interest in improving this project. Please read this document before opening an issue or pull request.

## Ground Rules

- **No GitHub Actions.** This repository intentionally does not enable Actions. Please do not open PRs that add `.github/workflows/` files.
- **No external CI integrations.** The maintainers do not currently accept PRs that connect this repository to Jenkins, CircleCI, Travis, or other services.
- **One PR, one purpose.** Separate functional changes from formatting / refactors.
- **Conventional Commits.** Use types like `feat:`, `fix:`, `docs:`, `refactor:`, `chore:` in commit subjects.

## Development

Required on the contributor's machine:

- `bash` >= 4
- `shellcheck` (for local linting)
- `git`

Before pushing a change to the script:

```bash
# Syntax check
bash -n mlperf.sh

# Static analysis
shellcheck mlperf.sh
```

## Pull Request Workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your change. Keep commits focused and well-described.
3. Ensure `bash -n` passes and `shellcheck` produces no new warnings.
4. Open a pull request against `master`.
5. At least one approving review from a CODEOWNER is required to merge.
6. Force pushes to `master` are prohibited.
7. Commits to `master` must be GPG- or SSH-signed.

## Adding a New Prompt or Mode

If your change introduces a new user prompt:

- Use the existing `ask` / `ask_req` / `yesno` / `pick` helpers — do not call `read` directly.
- Validate user input with `validate_path` if it is a filesystem path.
- Add the new option to the relevant `OPTS` / `LAUNCH_OPTS` array in the correct phase.
- Update `README.md` section 8 (Runtime Matrix) if the change adds or removes a launcher.
- Document any new edge case in `README.md` section 10.

## Reporting Bugs

Open a [GitHub issue](https://github.com/DoNnMyTh/mlperf/issues/new) and include:

- OS, shell, Bash version (`bash --version`).
- Exact command you ran.
- Exact prompts and answers (or a transcript).
- Full error output, including at least 20 lines of surrounding context.

## Reporting Security Vulnerabilities

See [SECURITY.md](SECURITY.md). **Do not** report security issues in public GitHub issues.

## Licensing

By submitting a pull request, you agree that your contribution will be licensed under the MIT License (see [LICENSE](LICENSE)).
