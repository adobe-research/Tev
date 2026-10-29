#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_tb4_cm.sh — Terminal-Bench 4.0, computing_math (CM) subset (k=5), on ONE 8-GPU node:
#   Harbor 0.20.0 venv -> vLLM server (docker) -> harbor run over the CM tasks -> pass@k stats.
#
# Dataset terminal-bench/terminal-bench@4.0.0, the paper's 18 CM tasks (tb_cm/tb4_cm.txt), agent timeout x1
# (TB 4.0 tasks already carry long agent timeouts). The task list also documents (commented out) the
# in-domain tasks the paper leaves out because a CPU-only Docker sandbox cannot run and grade them. A task
# tagged '# gpu' is skipped unless INCLUDE_GPU_TASKS=1, for a Harbor environment (HARBOR_ENV) that can provide
# sandbox GPUs (Harbor's docker backend cannot, and a GPU task would abort the whole run).
#
# Agent: Terminus-2 (Terminal-Bench's reference agent) against the local endpoint, temperature 0.7,
# top_p 0.95, <=32768 tokens per call, thinking on. Serving (Qwen3.8-27B, serve_qwen38_vllm.sh):
# vllm/vllm-openai:qwen38-x86_64-cu129, DP=8 x TP=1, 131072 ctx, bf16 KV, max 256 seqs per replica.
# LM_ONLY=1 (default) serves --language_model_only (text model only): required for RL exports, and what we
# use for the released model too.
#
# Usage:
#   MODEL_PATH=/path/to/Tev-27B bash scripts/eval/eval_tb4_cm.sh
# (MODEL_PATH = an HF dir, e.g. from scripts/eval/convert_ckpt_to_hf.sh.)
#
# Results: harbor trials under jobs/$JOB_NAME/ (repo root); metrics in $RESULTS_DIR/{metrics.json,stats.txt,
# metrics_clean.json,stats_clean.txt}. Re-running with the same JOB_NAME and settings resumes the harbor job
# (harbor refuses to resume if the config changed); with RETRY_INFRA=1 trials that died of serving/API
# errors are deleted first so they re-run instead of counting as failures.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=/path/to/hf/<model>}"
SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL_PATH%/}")}"
JOB_NAME="${JOB_NAME:-tb4-cm-$SERVED_NAME}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$JOB_NAME}"
DATASET="${DATASET:-terminal-bench/terminal-bench@4.0.0}"
LOCAL_DATASET="${LOCAL_DATASET:-}"                    # optional local tasks dir (-p) instead of the Hub dataset
TASKS_FILE="${TASKS_FILE:-$REPO_ROOT/scripts/eval/tb_cm/tb4_cm.txt}"
INCLUDE_GPU_TASKS="${INCLUDE_GPU_TASKS:-0}"
ENV_BACKEND="${HARBOR_ENV:-docker}"
K="${K:-5}"
N_CONCURRENT="${N_CONCURRENT:-16}"
MAX_RETRIES="${MAX_RETRIES:-2}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-32768}"                     # per model call; thinking needs room to finish + emit a command
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
CTX_INPUT_TOKENS="${CTX_INPUT_TOKENS:-$(( MAX_MODEL_LEN - MAX_TOKENS - 8192 ))}"   # agent's context budget; 0 = don't register
MAX_TURNS="${MAX_TURNS:-}"                            # empty = Terminus-2 default
THINKING="${THINKING:-on}"                            # off -> chat_template_kwargs.enable_thinking=false
PORT="${PORT:-8008}"
KEEP_SERVER="${KEEP_SERVER:-0}"
RETRY_INFRA="${RETRY_INFRA:-1}"
# (grader hangs, VerifierTimeoutError, are not re-queued: they would just hang again)
RETRY_INFRA_EXC="${RETRY_INFRA_EXC:-UnknownApiError APIError APIConnectionError APITimeoutError ServiceUnavailable InternalServerError}"
export WORK="${WORK:-$HOME/.cache/tev_eval}"
export HARBOR21_VENV="${HARBOR21_VENV:-$WORK/.venv-harbor21}"
HARBOR_BIN="$HARBOR21_VENV/bin/harbor"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
export DOCKER_SH
export LM_ONLY="${LM_ONLY:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"   # dummy; the local vLLM has no auth

say() { echo "[eval_tb4_cm $(date -u +%FT%TZ)] $*"; }
die() { say "FATAL: $*"; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || die "no config.json in MODEL_PATH=$MODEL_PATH"
ls "$MODEL_PATH"/*.safetensors >/dev/null 2>&1 || die "no *.safetensors in $MODEL_PATH"
[ -f "$TASKS_FILE" ] || die "no task list $TASKS_FILE"
mkdir -p "$RESULTS_DIR"
CONTAINER="vllm-$SERVED_NAME"

# task selection: one -i per task (Hub task ids are terminal-bench/<name>; local ones are the dir name)
if [ -n "$LOCAL_DATASET" ]; then DS_ARGS=(-p "$LOCAL_DATASET"); TASK_PREFIX=""; else DS_ARGS=(-d "$DATASET"); TASK_PREFIX="terminal-bench/"; fi
TASK_ARGS=(); TASKS=(); SKIPPED=()
while read -r name tag; do
  if [ "$tag" = "gpu" ] && [ "$INCLUDE_GPU_TASKS" != "1" ]; then SKIPPED+=("$name"); continue; fi
  TASKS+=("$name"); TASK_ARGS+=(-i "$TASK_PREFIX$name")
done < <(sed -e 's/#[[:space:]]*gpu.*/ gpu/' -e 's/#.*//' -e '/^[[:space:]]*$/d' "$TASKS_FILE")
[ "${#TASKS[@]}" -gt 0 ] || die "no tasks selected from $TASKS_FILE"

say "dataset=${LOCAL_DATASET:-$DATASET} tasks=${#TASKS[@]} model=$MODEL_PATH served_name=$SERVED_NAME k=$K n=$N_CONCURRENT job=$JOB_NAME"
[ "${#SKIPPED[@]}" = 0 ] || say "skipping GPU task(s) (INCLUDE_GPU_TASKS=1 to run): ${SKIPPED[*]}"

# 1. isolated Harbor 0.20.0 venv + docker compose plugin
bash scripts/eval/setup_harbor21.sh || die "setup_harbor21"

# 2. serve + wait (big models can take a while to load)
MODEL="$MODEL_PATH" SERVED_NAME="$SERVED_NAME" PORT="$PORT" VLLM_CONTAINER="$CONTAINER" \
  VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-x86_64-cu129}" \
  DP="${DP:-8}" TP="${TP:-1}" MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-none}" ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}" \
  SPEC_DECODING="${SPEC_DECODING:-0}" \
  bash scripts/eval/serve_qwen38_vllm.sh || die "vLLM serve launch"
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

# 3. resume hygiene: re-queue trials that failed for infrastructure reasons (not the model's fault)
if [ "$RETRY_INFRA" = "1" ] && [ -d "jobs/$JOB_NAME" ]; then
  python3 - "jobs/$JOB_NAME" $RETRY_INFRA_EXC <<'PY'
import json, os, shutil, sys
job, excs = sys.argv[1], sys.argv[2:]
for t in sorted(os.listdir(job)):
    p = os.path.join(job, t, "result.json")
    if not os.path.isfile(p):
        continue
    try:
        e = (json.load(open(p)).get("exception_info") or {}).get("exception_type") or ""
    except Exception:
        continue
    if any(x in e for x in excs):
        print(f"re-queue {t} ({e})"); shutil.rmtree(os.path.join(job, t))
PY
fi

# 4. run the CM subset with Terminus-2
AGENT_ARGS=(
  -a terminus-2
  -m "openai/$SERVED_NAME"
  --ak "api_base=http://localhost:$PORT/v1"
  --ak "temperature=$TEMPERATURE"
  --ak "top_p=$TOP_P"
  --ak "max_tokens=$MAX_TOKENS"
)
# register the served context window with litellm (otherwise the agent assumes a generic default)
if [ "$CTX_INPUT_TOKENS" -gt 0 ] 2>/dev/null; then
  AGENT_ARGS+=(--ak "model_info={\"max_input_tokens\":$CTX_INPUT_TOKENS,\"max_output_tokens\":$MAX_TOKENS,\"input_cost_per_token\":0,\"output_cost_per_token\":0,\"litellm_provider\":\"openai\",\"mode\":\"chat\"}")
fi
[ -n "$MAX_TURNS" ] && AGENT_ARGS+=(--ak "max_turns=$MAX_TURNS")
[ "$THINKING" = "off" ] && AGENT_ARGS+=(--ak 'llm_call_kwargs={"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}')
CMD=( "$HARBOR_BIN" run
      "${DS_ARGS[@]}"
      -e "$ENV_BACKEND"
      "${AGENT_ARGS[@]}"
      "${TASK_ARGS[@]}"
      -n "$N_CONCURRENT"
      -k "$K"
      -r "$MAX_RETRIES"
      --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT"
      -o jobs
      --job-name "$JOB_NAME"
      --yes )
CMD_STR="OPENAI_API_KEY=$(printf '%q' "$OPENAI_API_KEY")"
for a in "${CMD[@]}"; do CMD_STR+=" $(printf '%q' "$a")"; done
$DOCKER_SH "$CMD_STR" || say "WARN: harbor returned nonzero (continuing to stats)"

# 5. stats
"$HARBOR21_VENV/bin/python" scripts/eval/compute_stats.py "jobs/$JOB_NAME" \
  --json-output "$RESULTS_DIR/metrics.json" | tee "$RESULTS_DIR/stats.txt" || die "compute_stats"
"$HARBOR21_VENV/bin/python" scripts/eval/compute_stats_clean.py "jobs/$JOB_NAME" -k "$K" \
  --json-output "$RESULTS_DIR/metrics_clean.json" > "$RESULTS_DIR/stats_clean.txt" 2>&1 || say "WARN: compute_stats_clean failed"
python3 - "$RESULTS_DIR/metrics.json" "${#TASKS[@]}" "$K" <<'PY' || true
import json, sys
m, n, k = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
short = [t for t, v in m["per_task"].items() if v["n_attempts"] < k]
if m["n_tasks"] != n or short:
    print(f"[eval_tb4_cm] WARN: incomplete: {m['n_tasks']}/{n} tasks have results; <{k} trials: {short}")
PY
say "DONE: $RESULTS_DIR/metrics.json"
