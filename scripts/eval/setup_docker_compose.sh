#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_docker_compose.sh — install the Docker Compose v2 CLI plugin at USER
# level (~/.docker/cli-plugins/docker-compose).
#
# WHY: Harbor's `--env docker` backend shells out to `docker compose ... up`. Hosts that ship dockerd
# without the compose plugin fail every trial with "unknown flag: --project-name" (rc 125).
# User-level install => no sudo, no system change. Idempotent.
#
# Install order: already available -> COMPOSE_CACHE (optional pre-downloaded binary, network-free) ->
# pinned GitHub release download with retries; fails loudly if all fail.
# ---------------------------------------------------------------------------
set -uo pipefail
DEST="$HOME/.docker/cli-plugins"
mkdir -p "$DEST"

CACHE="${COMPOSE_CACHE:-}"
VER="${COMPOSE_VERSION:-v5.4.0}"
URL="https://github.com/docker/compose/releases/download/${VER}/docker-compose-linux-x86_64"

DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
_verify() { $DOCKER_SH 'docker compose version' >/dev/null 2>&1; }
_install() {  # $1 = source binary path
    cp -f "$1" "$DEST/docker-compose" && chmod +x "$DEST/docker-compose" && _verify
}

# 0. already available (image or a prior step)?
if _verify; then
    echo "[compose] already available: $($DOCKER_SH 'docker compose version' 2>/dev/null | head -1)"
    exit 0
fi

# 1. durable cache — network-free, preferred path.
if [ -n "$CACHE" ] && [ -f "$CACHE" ]; then
    echo "[compose] installing from cache $CACHE -> $DEST/docker-compose"
    if _install "$CACHE"; then
        echo "[compose] ok (cache): $($DOCKER_SH 'docker compose version' 2>/dev/null | head -1)"
        exit 0
    fi
    echo "[compose] WARN: cache install/verify failed; falling back to download" >&2
else
    echo "[compose] cache $CACHE not present; falling back to download" >&2
fi

# 2. pinned github download with retries — fallback only.
TMP="$(mktemp)"
ok=0
for i in 1 2 3 4 5; do
    echo "[compose] download attempt $i/5: $VER"
    if curl -fsSL --retry 3 --connect-timeout 20 "$URL" -o "$TMP"; then ok=1; break; fi
    sleep $((i * 5))
done
if [ "$ok" != 1 ]; then
    echo "ERROR: could not download docker compose $VER from $URL (github egress/rate-limit?). Pre-download it and set COMPOSE_CACHE=<path>." >&2
    rm -f "$TMP"; exit 1
fi
if _install "$TMP"; then
    echo "[compose] ok (downloaded $VER): $($DOCKER_SH 'docker compose version' 2>/dev/null | head -1)"
    rm -f "$TMP"; exit 0
fi
echo "ERROR: docker compose install/verify failed after downloading $VER" >&2
rm -f "$TMP"; exit 1
