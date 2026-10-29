#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_harbor21_agent.sh — run TERMINAL-BENCH 2.1 against a local vLLM endpoint
# with an ARBITRARY harbor agent, using the ISOLATED modern Harbor venv built by
# setup_harbor21.sh. Generalization of run_harbor21_eval.sh (which is hard-wired
# to Vanillux2Agent + its own --ak set, used by the 9B evals).
#
# eval_tb21.sh's 27B preset uses it with Terminus-2. Any harbor agent works against the
# same vLLM serving, e.g.:
#   * AGENT=terminus-2                     -> TB's reference Terminus-2 agent
#   * AGENT=Vanillux2Agent:Vanillux2Agent  -> the in-repo training harness
#   * AGENT=oracle                          -> built-in smoke, no model
#
# Sampling/thinking are governed by the model's shipped generation_config.json
# (temperature=1.0, top_p=0.95, top_k=20) on the vLLM side; per-agent knobs such
# as reasoning_effort / temperature are passed through AGENT_AK.
#
# Env:
#   AGENT          harbor -a value or 'module:Class' import path (default oracle)
#   MODEL_NAME     vLLM --served-model-name (litellm model becomes openai/<name>)
#   API_BASE       vLLM OpenAI endpoint (default http://localhost:8008/v1)
#   DATASET        Hub org/name (default terminal-bench/terminal-bench-2-1)
#   LOCAL_DATASET  optional local export dir -> use -p instead of -d
#   K N_CONCURRENT JOB_NAME JOBS_DIR
#   AGENT_AK       space-separated extra "key=value" agent kwargs (--ak each)
#   AGENT_ENV      space-separated "KEY=value" env vars to pass to the agent (--ae each)
#   TIMEOUT_MULT / AGENT_TIMEOUT_MULT  harbor timeout multipliers (default 1.0 / unset)
#   MODEL_PREFIX   litellm provider prefix for the model (default openai/)
#
# Runs harbor via $DOCKER_SH (default `bash -c`; `sg docker -c` for a stale docker group) with the repo on
# PYTHONPATH so custom agents (Vanillux2Agent) import from source.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WORK="${WORK:-$HOME/.cache/tev_eval}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
HARBOR21_VENV="${HARBOR21_VENV:-$WORK/.venv-harbor21}"
HARBOR_BIN="$HARBOR21_VENV/bin/harbor"
[ -x "$HARBOR_BIN" ] || { echo "ERROR: $HARBOR_BIN missing — run setup_harbor21.sh first" >&2; exit 1; }

DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"
LOCAL_DATASET="${LOCAL_DATASET:-}"
AGENT="${AGENT:-oracle}"
ENV_BACKEND="${HARBOR_ENV:-docker}"
MODEL_NAME="${MODEL_NAME:-qwen38-27b}"
MODEL_PREFIX="${MODEL_PREFIX:-openai/}"
API_BASE="${API_BASE:-http://localhost:8008/v1}"
JOB_NAME="${JOB_NAME:-${MODEL_NAME}-tb21}"
K="${K:-1}"
N_CONCURRENT="${N_CONCURRENT:-8}"
JOBS_DIR="${JOBS_DIR:-jobs}"
TIMEOUT_MULT="${TIMEOUT_MULT:-1.0}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-}"

# litellm talks to an OpenAI-compatible endpoint; keep a non-empty (dummy) key so
# the openai/ provider does not error on a missing key.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Dataset selector: local export (-p) if given, else Hub (-d).
if [ -n "$LOCAL_DATASET" ]; then DS_ARGS="-p '${LOCAL_DATASET}'"; else DS_ARGS="-d '${DATASET}'"; fi

# Assemble agent args.
if [ "$AGENT" = "oracle" ]; then
  AGENT_ARGS="-a oracle"
else
  AGENT_ARGS="-a '${AGENT}' -m '${MODEL_PREFIX}${MODEL_NAME}' --ak api_base='${API_BASE}'"
  for kv in ${AGENT_AK:-}; do AGENT_ARGS="$AGENT_ARGS --ak ${kv}"; done
  for kv in ${AGENT_ENV:-}; do AGENT_ARGS="$AGENT_ARGS --ae ${kv}"; done
fi

TO_ARGS="--timeout-multiplier ${TIMEOUT_MULT}"
[ -n "$AGENT_TIMEOUT_MULT" ] && TO_ARGS="$TO_ARGS --agent-timeout-multiplier ${AGENT_TIMEOUT_MULT}"

LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="-l ${LIMIT}"

echo "[harbor21-agent] dataset=${LOCAL_DATASET:-$DATASET} agent=$AGENT model=${MODEL_PREFIX}${MODEL_NAME} env=$ENV_BACKEND k=$K n=$N_CONCURRENT job=$JOB_NAME ${LIMIT:+limit=$LIMIT}"
echo "[harbor21-agent] extra_ak='${AGENT_AK:-}' timeout_mult=$TIMEOUT_MULT agent_timeout_mult=${AGENT_TIMEOUT_MULT:-default}"
echo "[harbor21-agent] harbor=$("$HARBOR_BIN" --version 2>&1 | head -1)"

$DOCKER_SH "PYTHONPATH='$REPO_ROOT' OPENAI_API_KEY='$OPENAI_API_KEY' '$HARBOR_BIN' run \
  ${DS_ARGS} \
  -e ${ENV_BACKEND} \
  ${AGENT_ARGS} \
  -n ${N_CONCURRENT} \
  -k ${K} ${LIMIT_ARG} \
  ${TO_ARGS} \
  -o '${JOBS_DIR}' \
  --job-name '${JOB_NAME}'"

echo "[harbor21-agent] done -> ${JOBS_DIR}/${JOB_NAME}/"
