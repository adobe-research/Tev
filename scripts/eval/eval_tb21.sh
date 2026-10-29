#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_tb21.sh — Terminal-Bench 2.1 (k=5) for a trained checkpoint on ONE 8-GPU node:
#   Harbor 0.20.0 venv -> vLLM server (docker) -> harbor run -> pass@k stats.
#
# Presets (the configurations we report):
#   PRESET=9b   Qwen3.5-9B exports: vllm/vllm-openai:v0.19.1, DP=8 x TP=1, bf16, qwen3_xml tool parser,
#               agent = Vanillux2Agent (this repo; same bash-tool harness as training), 16 concurrent trials.
#   PRESET=27b  Qwen3.8-27B exports: vllm/vllm-openai:qwen38-x86_64-cu129 (needs vLLM>=0.17 + transformers>=5.8),
#               DP=4 x TP=2, 262144 ctx, bf16 KV, prefix caching, agent = Terminus-2 (reasoning_effort=xhigh,
#               temperature 1.0), agent timeout x3, 2 concurrent trials.
# Both serve with --language_model_only (LM_ONLY=1): RL exports carry only the text model. For an untrained
# base model set LM_ONLY=0 (and, for Qwen3.8, SPEC_DECODING=1 to use its MTP head).
#
# Usage:
#   MODEL_PATH=/path/to/hf/tev-9b bash scripts/eval/eval_tb21.sh
#   PRESET=27b MODEL_PATH=/path/to/hf/tev-27b bash scripts/eval/eval_tb21.sh
# (MODEL_PATH = an HF dir, e.g. from scripts/eval/convert_ckpt_to_hf.sh.)
#
# Results: harbor trials under jobs/$JOB_NAME/ (repo root); metrics in $RESULTS_DIR/{metrics.json,stats.txt,
# metrics_clean.json,stats_clean.txt}. Re-running with the same JOB_NAME resumes the harbor job; with
# RETRY_INFRA=1 trials that died of infrastructure errors are deleted first so they re-run instead of
# counting as failures.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PRESET="${PRESET:-9b}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=/path/to/hf/<model>}"
SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL_PATH%/}")}"
JOB_NAME="${JOB_NAME:-tb21-$SERVED_NAME}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$JOB_NAME}"
DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"
K="${K:-5}"
PORT="${PORT:-8008}"
KEEP_SERVER="${KEEP_SERVER:-0}"
RETRY_INFRA="${RETRY_INFRA:-1}"
RETRY_INFRA_EXC="${RETRY_INFRA_EXC:-UnknownApiError VerifierTimeoutError APIError APIConnectionError APITimeoutError ServiceUnavailable InternalServerError}"
export WORK="${WORK:-$HOME/.cache/tev_eval}"
export HARBOR21_VENV="${HARBOR21_VENV:-$WORK/.venv-harbor21}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
export DOCKER_SH
export LM_ONLY="${LM_ONLY:-1}"

say() { echo "[eval_tb21 $(date -u +%FT%TZ)] $*"; }
die() { say "FATAL: $*"; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || die "no config.json in MODEL_PATH=$MODEL_PATH"
ls "$MODEL_PATH"/*.safetensors >/dev/null 2>&1 || die "no *.safetensors in $MODEL_PATH"
mkdir -p "$RESULTS_DIR"
CONTAINER="vllm-$SERVED_NAME"

case "$PRESET" in
  9b)
    N_CONCURRENT="${N_CONCURRENT:-16}"
    serve() {
      MODEL="$MODEL_PATH" SERVED_NAME="$SERVED_NAME" PORT="$PORT" DP="${DP:-8}" TP="${TP:-1}" DTYPE="${DTYPE:-bfloat16}" \
        PARSER="${PARSER:-qwen3_xml}" VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.1}" VLLM_CONTAINER="$CONTAINER" \
        bash scripts/eval/serve_vllm.sh
    }
    run_harbor() {
      K="$K" N_CONCURRENT="$N_CONCURRENT" MODEL_NAME="$SERVED_NAME" API_BASE="http://localhost:$PORT/v1" \
        JOB_NAME="$JOB_NAME" DATASET="$DATASET" bash scripts/eval/run_harbor21_eval.sh
    }
    ;;
  27b)
    N_CONCURRENT="${N_CONCURRENT:-2}"
    serve() {
      MODEL="$MODEL_PATH" SERVED_NAME="$SERVED_NAME" PORT="$PORT" VLLM_CONTAINER="$CONTAINER" \
        VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-x86_64-cu129}" \
        DP="${DP:-4}" TP="${TP:-2}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}" KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-none}" \
        ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}" SPEC_DECODING="${SPEC_DECODING:-0}" \
        bash scripts/eval/serve_qwen38_vllm.sh
    }
    run_harbor() {
      AGENT="terminus-2" AGENT_AK="reasoning_effort=${REASONING_EFFORT:-xhigh} temperature=1.0" \
        MODEL_NAME="$SERVED_NAME" API_BASE="http://localhost:$PORT/v1" DATASET="$DATASET" K="$K" \
        N_CONCURRENT="$N_CONCURRENT" JOB_NAME="$JOB_NAME" TIMEOUT_MULT="${TIMEOUT_MULT:-1.0}" \
        AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-3}" bash scripts/eval/run_harbor21_agent.sh
    }
    ;;
  *) die "PRESET must be 9b or 27b" ;;
esac

say "preset=$PRESET model=$MODEL_PATH served_name=$SERVED_NAME k=$K n=$N_CONCURRENT job=$JOB_NAME"

# 1. isolated Harbor 0.20.0 venv + docker compose plugin
bash scripts/eval/setup_harbor21.sh || die "setup_harbor21"

# 2. serve + wait (big models can take a while to load)
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

# 4. run Terminal-Bench 2.1
run_harbor || say "WARN: harbor returned nonzero (continuing to stats)"

# 5. stats
"$HARBOR21_VENV/bin/python" scripts/eval/compute_stats.py "jobs/$JOB_NAME" \
  --json-output "$RESULTS_DIR/metrics.json" | tee "$RESULTS_DIR/stats.txt" || die "compute_stats"
"$HARBOR21_VENV/bin/python" scripts/eval/compute_stats_clean.py "jobs/$JOB_NAME" -k "$K" \
  --json-output "$RESULTS_DIR/metrics_clean.json" > "$RESULTS_DIR/stats_clean.txt" 2>&1 || say "WARN: compute_stats_clean failed"
say "DONE: $RESULTS_DIR/metrics.json"
