#!/usr/bin/env bash
# Push a prepared MLCommons submission tree as a pull request.
#
# Given the output of tools/submit.sh (a directory containing <submitter>/),
# this tool:
#   1. Forks mlcommons/training_results_vX.Y to the authenticated gh user
#      (if not already forked).
#   2. Clones the fork into a scratch dir.
#   3. Copies the submission tree to <submitter>/ in the fork.
#   4. Creates a branch submission/<workload>/<system_desc>/<timestamp>.
#   5. Commits and pushes.
#   6. Opens a pull request to mlcommons/training_results_vX.Y.
#
# Does not bypass the MLCommons submitter working group review; it only
# automates the mechanical PR step.

set -u
set -o pipefail

# --- mlperf.sh common-lib hook -----------------------------------------
_MLPERF_LIB_SOURCED=0
if _LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/common.sh" && [[ -f "$_LIB" ]]; then
    # shellcheck source=../lib/common.sh
    source "$_LIB"
    _MLPERF_LIB_SOURCED=1
fi
# Auto-yes / config-file via env only — no flag parsing here to avoid
# clobbering per-tool argv handling.
: "${MLPERF_AUTO_YES:=0}"
if [[ -n "${MLPERF_CONFIG_FILE:-}" && -f "${MLPERF_CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${MLPERF_CONFIG_FILE}"
    MLPERF_AUTO_YES=1
fi
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

(( BASH_VERSINFO[0] >= 4 )) || die "Bash >= 4 required"
[[ -t 0 ]] || die "TTY required"

command -v gh >/dev/null 2>&1   || die "gh CLI required (https://cli.github.com/)"
command -v git >/dev/null 2>&1  || die "git required"
gh auth status >/dev/null 2>&1  || die "Not authenticated. Run: gh auth login"

# -----------------------------------------------------------------
SUB_DIR="$(ask_req 'Path to prepared submission directory (output of tools/submit.sh)')"
[[ -d "$SUB_DIR" ]] || die "Not a directory: $SUB_DIR"
mapfile -t SUBMITTERS < <(find "$SUB_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n')
case "${#SUBMITTERS[@]}" in
    0) die "No submitter subtree in $SUB_DIR" ;;
    1) SUBMITTER="${SUBMITTERS[0]}" ;;
    *) err "Multiple submitter subtrees:"; printf '  %s\n' "${SUBMITTERS[@]}" >&2
       SUBMITTER="$(ask_req 'Which one to submit')"
       [[ -d "$SUB_DIR/$SUBMITTER" ]] || die "Not found: $SUB_DIR/$SUBMITTER" ;;
esac
info "Submitter: $SUBMITTER"

UPSTREAM="$(ask 'Upstream repo to PR against' 'mlcommons/training_results_v5.1')"
UPSTREAM_BRANCH="$(ask 'Upstream base branch' 'main')"

# -----------------------------------------------------------------
# Identify a workload + system from the tree (used for branch/PR title).
WL=""
for d in "$SUB_DIR/$SUBMITTER/benchmarks/"*/; do
    [[ -d "$d" ]] && { WL="$(basename "$d")"; break; }
done
SYS=""
for d in "$SUB_DIR/$SUBMITTER/results/"*/; do
    [[ -d "$d" ]] && { SYS="$(basename "$d")"; break; }
done
[[ -z "$WL" || -z "$SYS" ]] && die "Could not determine workload/system from $SUB_DIR"
info "Workload: $WL   System: $SYS"

BRANCH="submission/${WL}/${SYS}/$(date +%Y%m%d-%H%M%S)"
WORK="$(mktemp -d)"
info "Scratch: $WORK"
# Trap so the multi-GB fork clone does not leak under /tmp on abort.
trap 'rm -rf "$WORK"' EXIT
trap 'err "aborted"; rm -rf "$WORK"; exit 130' INT TERM

# -----------------------------------------------------------------
# Fork + clone
GH_USER="$(gh api user --jq .login)"
FORK="$GH_USER/$(basename "$UPSTREAM")"
info "gh user: $GH_USER   fork: $FORK"

if ! gh repo view "$FORK" >/dev/null 2>&1; then
    yesno "Fork $UPSTREAM to $FORK now?" y || die "Cannot proceed without a fork."
    gh repo fork "$UPSTREAM" --clone=false --remote=false || die "fork failed"
    # Wait for fork to become available
    for _ in $(seq 1 30); do
        gh repo view "$FORK" >/dev/null 2>&1 && break
        sleep 2
    done
fi

say "Cloning fork"
gh repo clone "$FORK" "$WORK/fork" -- --branch "$UPSTREAM_BRANCH" || die "clone failed"
cd "$WORK/fork"
git remote add upstream "https://github.com/$UPSTREAM.git" 2>/dev/null || true
# Fetch upstream base branch; abort if it is unreachable — creating the branch
# from the fork's (possibly stale) HEAD would put unrelated commits in the PR.
git fetch upstream "$UPSTREAM_BRANCH" --depth=1 \
    || die "Failed to fetch upstream/$UPSTREAM_BRANCH. Check network / repo access."
git checkout -B "$BRANCH" "upstream/$UPSTREAM_BRANCH" \
    || die "Failed to branch from upstream/$UPSTREAM_BRANCH."

say "Copying submission tree"
mkdir -p "$SUBMITTER"
cp -r "$SUB_DIR/$SUBMITTER/." "$SUBMITTER/"

git add "$SUBMITTER"
COMMIT_MSG="${SUBMITTER}: ${WL} on ${SYS} (MLPerf Training v5.1 submission)"
git -c user.name="$GH_USER" -c user.email="${GH_USER}@users.noreply.github.com" \
    commit -m "$COMMIT_MSG" || die "Nothing to commit"

say "Pushing branch $BRANCH to $FORK"
git push -u origin "$BRANCH" || die "push failed"

say "Opening pull request"
PR_BODY="$(cat <<EOF
## Submitter

$SUBMITTER

## Workload

$WL

## System

$SYS

## Summary

This PR adds the MLPerf Training v5.1 submission for **$WL** on **$SYS**.

### Files included

- Implementation source under \`$SUBMITTER/benchmarks/$WL/implementations/\`
- Result logs under \`$SUBMITTER/results/$SYS/$WL/\`
- Compliance-checker log under \`$SUBMITTER/results/$SYS/$WL/compliance_checker_log.txt\`
- System descriptor \`$SUBMITTER/systems/$SYS.json\`

### Local verification

All result logs were validated with:

\`\`\`
python -m mlperf_logging.compliance_checker --usage training --ruleset 5.1.0 <log>
\`\`\`

See \`compliance_checker_log.txt\` for the combined output.

---

Generated via https://github.com/DoNnMyTh/mlperf/blob/master/tools/submit_pr.sh
EOF
)"

PR_URL="$(gh pr create \
    --repo "$UPSTREAM" \
    --base "$UPSTREAM_BRANCH" \
    --head "${GH_USER}:${BRANCH}" \
    --title "$COMMIT_MSG" \
    --body "$PR_BODY" 2>&1)" || { err "$PR_URL"; die "gh pr create failed"; }

say "Pull request opened:"
info "  $PR_URL"
info "Next: work with the MLCommons submitter working group for review."
