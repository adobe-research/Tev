#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_ale_cm.sh — Agents' Last Exam (ALE) computing_math, 18 tasks x K=5 rollouts, on ONE 8-GPU node:
#   ALE checkout + venv + task data (setup_ale.sh) -> vLLM server (docker) -> K parallel `ale_run run`
#   -> pass@1 / pass@K over the 18 tasks.
#
# Agent: vanillux_oai (./vanillux_oai) = the harness the model is RL-trained with (one persistent `bash`
# tool, same prompts), driving the model over the local OpenAI-compatible endpoint. Every task is scored by
# its own ALE grader. Each rollout rN is an independent ale_run output root; the K roots run in parallel.
#
# Presets (the configurations we report):
#   PRESET=27b  Qwen3.8-27B: vllm/vllm-openai:qwen38-x86_64-cu129, DP=4 x TP=2, 65536 ctx, bf16 KV,
#               qwen3_coder tool parser (serve_qwen38_vllm.sh).
#   PRESET=9b   Qwen3.5-9B: vllm/vllm-openai:v0.19.1, DP=8 x TP=1, bf16, qwen3_xml tool parser (serve_vllm.sh).
# Both: temperature 0.7, top_p 0.95, thinking on, <=64 turns, <=16384 tokens/turn, 4 concurrent tasks per
# rollout, 2 h wall clock per task. LM_ONLY=1 (default) serves --language_model_only (text model only):
# required for RL exports, and what we use for the released model too.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.8-27B bash scripts/eval/ale_cm/eval_ale_cm.sh
#   PRESET=9b MODEL_PATH=/path/to/hf/<model> bash scripts/eval/ale_cm/eval_ale_cm.sh
# Prerequisite: access to the gated task-data dataset on Hugging Face + `hf auth login` (see setup_ale.sh).
#
# Results under $RESULTS_DIR (default results/$JOB_NAME): rollouts/r1..rK (ale_run outputs, one run.json per
# task), logs/, configs/ (the rendered ale_run yamls), stats.txt + metrics.json. Re-running with the same
# JOB_NAME resumes: tasks that already have a result in a rollout are not re-run.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

PRESET="${PRESET:-27b}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=/path/to/hf/<model>}"
SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL_PATH%/}")}"
JOB_NAME="${JOB_NAME:-ale-cm-$SERVED_NAME}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$JOB_NAME}"
TASKS_FILE="${TASKS_FILE:-$HERE/computing_math.txt}"
K="${K:-5}"
PORT="${PORT:-8008}"
CONCURRENCY="${CONCURRENCY:-4}"          # tasks in flight per rollout (so K*CONCURRENCY sandboxes at once)
WALL_TIME_S="${WALL_TIME_S:-7200}"       # per-task agent wall clock
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-16384}"        # per model call
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
EXECUTE_ALL_TOOL_CALLS="${EXECUTE_ALL_TOOL_CALLS:-false}"
KEEP_SERVER="${KEEP_SERVER:-0}"
export WORK="${WORK:-$HOME/.cache/tev_eval}"
export ALE_DIR="${ALE_DIR:-$WORK/agents-last-exam}"
export TASK_DATA="${TASK_DATA:-$ALE_DIR/task-data}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
export DOCKER_SH
export LM_ONLY="${LM_ONLY:-1}"

say() { echo "[eval_ale_cm $(date -u +%FT%TZ)] $*"; }
die() { say "FATAL: $*"; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || die "no config.json in MODEL_PATH=$MODEL_PATH"
ls "$MODEL_PATH"/*.safetensors >/dev/null 2>&1 || die "no *.safetensors in $MODEL_PATH"
[ -f "$TASKS_FILE" ] || die "no task list $TASKS_FILE"
case "$ENABLE_THINKING" in true|false) ;; *) die "ENABLE_THINKING must be true or false" ;; esac
case "$EXECUTE_ALL_TOOL_CALLS" in true|false) ;; *) die "EXECUTE_ALL_TOOL_CALLS must be true or false" ;; esac
CONTAINER="vllm-$SERVED_NAME"

case "$PRESET" in
  27b)
    serve() {
      MODEL="$MODEL_PATH" SERVED_NAME="$SERVED_NAME" PORT="$PORT" VLLM_CONTAINER="$CONTAINER" \
        VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-x86_64-cu129}" \
        DP="${DP:-4}" TP="${TP:-2}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-none}" \
        ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}" SPEC_DECODING="${SPEC_DECODING:-0}" \
        bash scripts/eval/serve_qwen38_vllm.sh
    }
    ;;
  9b)
    serve() {
      MODEL="$MODEL_PATH" SERVED_NAME="$SERVED_NAME" PORT="$PORT" DP="${DP:-8}" TP="${TP:-1}" DTYPE="${DTYPE:-bfloat16}" \
        PARSER="${PARSER:-qwen3_xml}" VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.1}" VLLM_CONTAINER="$CONTAINER" \
        bash scripts/eval/serve_vllm.sh
    }
    ;;
  *) die "PRESET must be 27b or 9b" ;;
esac

say "preset=$PRESET model=$MODEL_PATH served_name=$SERVED_NAME k=$K concurrency=$CONCURRENCY job=$JOB_NAME"
mkdir -p "$RESULTS_DIR"/{rollouts,logs,configs}

# 1. ALE checkout + agent + venv + task data + sandbox image
bash "$HERE/setup_ale.sh" || die "setup_ale"
ALE_PY="$ALE_DIR/.venv/bin/python"

# 2. the harness reads its endpoint from ALE's secret file
mkdir -p "$ALE_DIR/secret"
printf 'OPENAI_API_KEY=sk-local\nOPENAI_API_BASE=http://127.0.0.1:%s/v1\n' "$PORT" > "$ALE_DIR/secret/.env"

# 3. serve + wait (big models can take a while to load)
serve || die "vLLM serve launch"
ready=0
for _ in $(seq 1 720); do
  if curl -s -m 3 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; then ready=1; break; fi
  st="$($DOCKER_SH "docker inspect -f '{{.State.Status}}' $CONTAINER" 2>/dev/null)"
  if [ "$st" != "running" ]; then $DOCKER_SH "docker logs --tail 60 $CONTAINER" 2>&1 | tail -40; die "vLLM container exited"; fi
  sleep 5
done
[ "$ready" = 1 ] || die "vLLM not ready in time"
say "vLLM ready ($SERVED_NAME on :$PORT)"
[ "$KEEP_SERVER" = "1" ] || trap '$DOCKER_SH "docker rm -f $CONTAINER" >/dev/null 2>&1' EXIT

# 4. render the ale_run configs (absolute paths: ale_run resolves them relative to the yaml)
CFG="$RESULTS_DIR/configs"
sed -e "s|__SERVED_NAME__|$SERVED_NAME|g" -e "s|__SERVER_URL__|http://127.0.0.1:$PORT/v1|g" \
    -e "s|__MAX_TURNS__|$MAX_TURNS|g" -e "s|__MAX_TOKENS__|$MAX_TOKENS|g" \
    -e "s|__TEMPERATURE__|$TEMPERATURE|g" -e "s|__TOP_P__|$TOP_P|g" \
    -e "s|__ENABLE_THINKING__|$ENABLE_THINKING|g" -e "s|__EXECUTE_ALL_TOOL_CALLS__|$EXECUTE_ALL_TOOL_CALLS|g" \
    "$HERE/configs/agent_vanillux_oai.yaml" > "$CFG/agent.yaml"
sed -e "s|__TASK_DATA__|$(cd "$TASK_DATA" && pwd)|g" "$HERE/configs/env_docker.yaml" > "$CFG/env.yaml"
sed -e 's/#.*//' -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*$//' "$TASKS_FILE" > "$CFG/tasks.txt"
for i in $(seq 1 "$K"); do
  sed -e "s|__NAME__|${JOB_NAME}_r$i|g" -e "s|__SECRET_FILE__|$ALE_DIR/secret/.env|g" \
      -e "s|__AGENT_YAML__|$CFG/agent.yaml|g" -e "s|__ENV_YAML__|$CFG/env.yaml|g" \
      -e "s|__TASKS__|$CFG/tasks.txt|g" -e "s|__ROOT__|$RESULTS_DIR/rollouts/r$i|g" \
      -e "s|__CONCURRENCY__|$CONCURRENCY|g" -e "s|__WALL_TIME_S__|$WALL_TIME_S|g" \
      "$HERE/configs/exp.yaml" > "$CFG/exp_r$i.yaml"
done

# 5. K independent rollouts in parallel. Older ale_run needs --resume to skip finished tasks;
#    newer versions resume by default and have no such flag.
RESUME_FLAG=""
(cd "$ALE_DIR" && "$ALE_PY" -m ale_run run --help 2>/dev/null) | grep -q -- "--resume" && RESUME_FLAG="--resume"
say "running $K rollouts x $(wc -l < "$CFG/tasks.txt") tasks -> $RESULTS_DIR/rollouts/r1..r$K (logs: $RESULTS_DIR/logs/)"
pids=""
for i in $(seq 1 "$K"); do
  $DOCKER_SH "cd '$ALE_DIR' && '$ALE_PY' -m ale_run run '$CFG/exp_r$i.yaml' $RESUME_FLAG" \
    > "$RESULTS_DIR/logs/rollout_r$i.log" 2>&1 &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || say "WARN: some rollout returned nonzero (see $RESULTS_DIR/logs/; continuing to stats)"

# 6. stats
"$ALE_PY" "$HERE/summarize_ale_cm.py" "$RESULTS_DIR/rollouts" -k "$K" --tasks "$TASKS_FILE" \
  --json-output "$RESULTS_DIR/metrics.json" | tee "$RESULTS_DIR/stats.txt" || die "summarize_ale_cm"
say "DONE: $RESULTS_DIR/metrics.json"
