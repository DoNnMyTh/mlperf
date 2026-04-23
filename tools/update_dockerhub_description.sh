#!/usr/bin/env bash
# Push docs/DOCKER_HUB_DESCRIPTION.md as the Docker Hub long description for
# donnmyth/mlperf-nvidia. Docker Hub has no CLI for this — use the v2 API.
#
# Requires:
#   DOCKERHUB_USER      your hub username (default: donnmyth)
#   DOCKERHUB_TOKEN     a Personal Access Token with "Read, Write, Delete"
#                       scope from hub.docker.com/settings/security
#
# Usage:
#   DOCKERHUB_TOKEN=dckr_pat_xxx tools/update_dockerhub_description.sh

set -u
set -o pipefail

USER_="${DOCKERHUB_USER:-donnmyth}"
REPO="${DOCKERHUB_REPO:-mlperf-nvidia}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN env var required (PAT from hub.docker.com/settings/security)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESC_FILE="$SCRIPT_DIR/../docs/DOCKER_HUB_DESCRIPTION.md"
[[ -f "$DESC_FILE" ]] || { echo "missing: $DESC_FILE" >&2; exit 1; }

SHORT="MLPerf Training v5.1 reference images for RTX 40xx / H100 / H200 / B200 / GB200 — patched TransformerEngine arch list."

# Exchange PAT for a JWT (PATs don't work directly against /v2/repositories/).
JWT=$(curl -fsS -X POST \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USER_\",\"password\":\"$DOCKERHUB_TOKEN\"}" \
    https://hub.docker.com/v2/users/login/ | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')
[[ -n "$JWT" ]] || { echo "login failed" >&2; exit 1; }

# PATCH repository
python3 - <<PY
import json, os, urllib.request
body = json.dumps({
    "description": os.environ["SHORT"],
    "full_description": open(os.environ["DESC_FILE"]).read(),
}).encode()
req = urllib.request.Request(
    f"https://hub.docker.com/v2/repositories/{os.environ['USER_']}/{os.environ['REPO']}/",
    data=body, method="PATCH",
    headers={"Authorization": f"JWT {os.environ['JWT']}",
             "Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode()[:200])
PY
echo "OK"
