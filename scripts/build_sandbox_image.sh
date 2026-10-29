#!/usr/bin/env bash
# Build the agent sandbox image (docker/sandbox.Dockerfile), smoke-test it, and optionally save it as a
# `docker save` tarball and/or push it to a registry.
#
# The smoke test checks that no grader code is reachable in the image and that the agent toolchain works:
#   - no training_data_collection* file on /, `import training_data_collection` fails, PYTHONPATH unset
#   - pinned deps import at the pinned versions; cc compiles C (-c only: no libc6-dev, so it cannot link);
#     readelf and sqlite3 run
#   - default user is root, python is 3.13
#
#   bash scripts/build_sandbox_image.sh                                         # build + smoke -> tev-sandbox:latest
#   SAVE_TAR=/path/to/tev-sandbox.tar bash scripts/build_sandbox_image.sh      # + docker save
#   PUSH=1 REGISTRY=<your-registry> bash scripts/build_sandbox_image.sh        # + push <your-registry>/tev-sandbox:<tag>
#
# The example dataset (examples/data/) names `tev-sandbox:latest`, so a locally built image works as-is on a
# single node; for multi-node runs push it (or `docker load` the tarball on every node).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$HERE/../docker"

DOCKER="${DOCKER:-bash -c}"          # set DOCKER='sg docker -c' if your docker group membership is stale
TAG="${IMAGE_TAG:-latest}"
SAVE_TAR="${SAVE_TAR:-}"
PUSH="${PUSH:-0}"
LOCAL_TAG="tev-sandbox:$TAG"

# --- 1. build from a context with NO codebase in it ---
CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT
cp "$DOCKER_DIR/sandbox.Dockerfile" "$CTX/Dockerfile"
cp "$DOCKER_DIR/requirements-sandbox.txt" "$CTX/requirements-sandbox.txt"
echo "=== build $LOCAL_TAG (context: $(ls "$CTX" | tr '\n' ' ')) ==="
$DOCKER "docker pull -q python:3.13-slim" >/dev/null
BASE_DIGEST="$($DOCKER "docker image inspect --format '{{index .RepoDigests 0}}' python:3.13-slim" | tr -d '\n')"
echo "base: $BASE_DIGEST"
$DOCKER "docker build -t $LOCAL_TAG -f $CTX/Dockerfile $CTX"

# --- 2. smoke test ---
echo "=== smoke test ==="
cat > "$CTX/smoke.sh" <<'SH'
set -u
fail() { echo "SMOKE FAIL: $*"; exit 1; }
[ "$(id -u)" = 0 ] || fail "default user is not root"
[ ! -e /opt/tdc ] || fail "/opt/tdc exists"
[ -z "${PYTHONPATH:-}" ] || fail "PYTHONPATH=$PYTHONPATH"
hit="$(find / -xdev \( -name 'training_data_collection*' -o -name 'public_validation*' -o -name '_PACKED_ORACLE*' \) -print -quit 2>/dev/null)"
[ -z "$hit" ] || fail "grader file present: $hit"
python3 -m training_data_collection.v72.families.cm.warehouse_etl_recovery.public_validation >/dev/null 2>&1 \
  && fail "advertised validation_command resolves"
python3 - <<'PY' || fail "python checks"
import importlib, importlib.util, sys
assert sys.version_info[:2] == (3, 13), sys.version
assert importlib.util.find_spec("training_data_collection") is None, "training_data_collection importable"
want = {l.split("==")[0]: l.split("==")[1] for l in open("/tmp/requirements-sandbox.txt").read().split() if "==" in l}
from importlib.metadata import version
bad = {k: (version(k), v) for k, v in want.items() if version(k) != v}
assert not bad, f"version mismatch: {bad}"
import numpy, safetensors, pandas, pydantic, yaml, cryptography, pytest
print("python", sys.version.split()[0], "| deps", " ".join(f"{k}={v}" for k, v in sorted(want.items())))
PY
cd /tmp && printf 'int f(int x){return x+1;}\n' > t.c && cc -std=c11 -O2 -c t.c -o t.o || fail "cc -c"
readelf -h /tmp/t.o | grep -q 'ELF64' || fail "readelf"
[ "$(echo "select 1+1;" | sqlite3)" = 2 ] || fail "sqlite3"
echo "cc -c, readelf, sqlite3 OK"
echo "SMOKE PASSED"
SH
$DOCKER "docker run --rm --network=none -v $CTX/smoke.sh:/smoke.sh:ro $LOCAL_TAG bash /smoke.sh"

IMAGE_ID="$($DOCKER "docker image inspect --format '{{.Id}}' $LOCAL_TAG" | tr -d '\n')"
echo "image id: $IMAGE_ID"

# --- 3. optional docker save -> local tarball ---
if [ -n "$SAVE_TAR" ]; then
  mkdir -p "$(dirname "$SAVE_TAR")"
  $DOCKER "docker save -o $SAVE_TAR $LOCAL_TAG"
  echo "=== saved: $SAVE_TAR ($(stat -c %s "$SAVE_TAR") bytes, id=$IMAGE_ID, base=$BASE_DIGEST) ==="
fi

# --- 4. optional registry push ---
if [ "$PUSH" = "1" ]; then
  REGISTRY="${REGISTRY:?set REGISTRY=<your-registry> (assumes docker login was done)}"
  DST="$REGISTRY/tev-sandbox:$TAG"
  echo "=== push -> $DST ==="
  $DOCKER "docker tag $LOCAL_TAG $DST"
  $DOCKER "docker push $DST"
  echo "    digest: $($DOCKER "docker inspect --format '{{index .RepoDigests 0}}' $DST" 2>/dev/null || echo '')"
fi
