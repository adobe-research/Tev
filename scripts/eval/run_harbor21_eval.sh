#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_harbor21_eval.sh — run TERMINAL-BENCH 2.1 against a local vLLM endpoint
# with the repo's Vanillux2Agent, using the ISOLATED modern Harbor venv built
# by setup_harbor21.sh (harbor 0.20.0). Dataset `terminal-bench/terminal-bench-2-1`
# (Hub) via -d, or a local export dir via -p (LOCAL_DATASET, e.g. an offline copy).
#
# Runs harbor via $DOCKER_SH (default `bash -c`; `sg docker -c` for a stale docker group) and exposes
# the repo on PYTHONPATH so `Vanillux2Agent` imports from source.
#
# Smoke (oracle, no model):  AGENT=oracle LIMIT=2 K=1 JOB_NAME=tb21-smoke \
#                              bash scripts/eval/run_harbor21_eval.sh
# Full run:                  K=5 N_CONCURRENT=16 MODEL_NAME=tev-9b \
#                              JOB_NAME=tb21-tev-9b bash scripts/eval/run_harbor21_eval.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WORK="${WORK:-$HOME/.cache/tev_eval}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
HARBOR21_VENV="${HARBOR21_VENV:-$WORK/.venv-harbor21}"
HARBOR_BIN="$HARBOR21_VENV/bin/harbor"
[ -x "$HARBOR_BIN" ] || { echo "ERROR: $HARBOR_BIN missing — run setup_harbor21.sh first" >&2; exit 1; }

DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"   # Hub org/name (2.1)
LOCAL_DATASET="${LOCAL_DATASET:-}"                          # optional: local export dir -> use -p
AGENT="${AGENT:-Vanillux2Agent:Vanillux2Agent}"            # or 'oracle' (built-in, no model)
ENV_BACKEND="${HARBOR_ENV:-docker}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<vLLM --served-model-name>}"  # vLLM --served-model-name
API_BASE="${API_BASE:-http://localhost:8008/v1}"
JOB_NAME="${JOB_NAME:-${MODEL_NAME}-tb21}"
K="${K:-5}"
N_CONCURRENT="${N_CONCURRENT:-16}"
MAX_FORMAT_ERRORS="${MAX_FORMAT_ERRORS:-64}"
JOBS_DIR="${JOBS_DIR:-jobs}"

# Dataset selector: local export (-p) if given, else Hub (-d).
if [ -n "$LOCAL_DATASET" ]; then DS_ARGS="-p '${LOCAL_DATASET}'"; else DS_ARGS="-d '${DATASET}'"; fi

LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="-l ${LIMIT}"

# Agent-specific args: oracle needs no model; custom agent calls our vLLM.
if [ "$AGENT" = "oracle" ]; then
  AGENT_ARGS="-a oracle"
else
  AGENT_ARGS="-a '${AGENT}' -m 'openai/${MODEL_NAME}' \
    --ak api_base='${API_BASE}' --ak max_format_errors=${MAX_FORMAT_ERRORS}"
fi

echo "[harbor21] dataset=${LOCAL_DATASET:-$DATASET} agent=$AGENT model=openai/$MODEL_NAME env=$ENV_BACKEND k=$K n=$N_CONCURRENT job=$JOB_NAME ${LIMIT:+limit=$LIMIT}"
echo "[harbor21] harbor=$("$HARBOR_BIN" --version 2>&1 | head -1)"

$DOCKER_SH "PYTHONPATH='$REPO_ROOT' '$HARBOR_BIN' run \
  ${DS_ARGS} \
  -e ${ENV_BACKEND} \
  ${AGENT_ARGS} \
  -n ${N_CONCURRENT} \
  -k ${K} ${LIMIT_ARG} \
  -o '${JOBS_DIR}' \
  --job-name '${JOB_NAME}'"

echo "[harbor21] done -> ${JOBS_DIR}/${JOB_NAME}/"
