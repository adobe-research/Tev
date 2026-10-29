#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_harbor21.sh — build an ISOLATED venv holding Harbor 0.20.0 for the Terminal-Bench 2.1 eval
# (`terminal-bench/terminal-bench-2-1` lives on the Harbor Hub registry, which needs a modern harbor).
#
# The repo's own agent (Vanillux2Agent) is pure-python and is made importable at run time via
# PYTHONPATH=<repo root> (see run_harbor21_eval.sh), so this venv only needs harbor + the agent's
# third-party deps: litellm (LLM calls), pyyaml (prompt templates), tqdm.
#
# Agent<->harbor ABI was verified against 0.20.0: harbor.agents.base.BaseAgent,
# harbor.environments.base.BaseEnvironment, harbor.models.agent.context.AgentContext
# all still exist and BaseAgent.run(instruction, environment, context) matches
# Vanillux2Agent unchanged.
#
# Idempotent: re-run cheaply; skips if the venv already has the right harbor.
#
# Usage:
#   bash scripts/eval/setup_harbor21.sh
#   HARBOR21_VERSION=0.20.0 HARBOR21_VENV=$HOME/.cache/tev_eval/.venv-harbor21 \
#     bash scripts/eval/setup_harbor21.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WORK="${WORK:-$HOME/.cache/tev_eval}"
HARBOR21_VERSION="${HARBOR21_VERSION:-0.20.0}"   # latest non-yanked (0.19.0/0.19.1 are yanked)
HARBOR21_VENV="${HARBOR21_VENV:-$WORK/.venv-harbor21}"

# Keep uv caches next to the venv (override UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR to reuse yours).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.uv-cache-local}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/.uv-python-local}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

HARBOR_BIN="$HARBOR21_VENV/bin/harbor"

have_version() {
  [ -x "$HARBOR_BIN" ] || return 1
  local v; v="$("$HARBOR_BIN" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  [ "$v" = "$HARBOR21_VERSION" ]
}

if have_version; then
  echo "[harbor21] venv already at harbor $HARBOR21_VERSION ($HARBOR21_VENV) — skip"
else
  echo "[harbor21] creating venv $HARBOR21_VENV (python 3.12) with harbor==$HARBOR21_VERSION"
  uv venv "$HARBOR21_VENV" --python 3.12
  uv pip install --python "$HARBOR21_VENV" \
    "harbor==$HARBOR21_VERSION" litellm tqdm pyyaml
fi

echo "[harbor21] harbor: $("$HARBOR_BIN" --version 2>&1 | head -1)  (bin=$HARBOR_BIN)"

# harbor's --env docker backend shells out to `docker compose`.
bash "$REPO_ROOT/scripts/eval/setup_docker_compose.sh"

echo "[harbor21] ready. Export HARBOR21_VENV=$HARBOR21_VENV for run_harbor21_eval.sh"
