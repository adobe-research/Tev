#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# serve_qwen38_vllm.sh — serve Qwen/Qwen3.8-27B (the OFFICIAL bf16 release) with
# the vLLM OpenAI docker image, using the flags from the official vLLM recipe
# (https://recipes.vllm.ai/Qwen/Qwen3.8-27B). Sibling of serve_vllm.sh (Qwen3.5 / 9B).
# Uses the prebuilt image (torch + CUDA + vLLM included), so there is no wheel install.
#
# Model facts driving these flags (from the released config.json / model card):
#   * model_type qwen3_5 (Qwen3_5ForConditionalGeneration) — hybrid linear+full
#     attention + Gated DeltaNet + MTP; a native VL (multimodal) THINKING model.
#     Requires vLLM >= 0.17.0 and transformers >= 5.8.0 (recipe); see the IMAGE note below.
#   * ~52 GB bf16 (27B). Served TP=2 x DP=4 on one 8xA100-80GB node with the full
#     262144 native context. This script defaults to the recipe's fp8 KV cache; the eval
#     launchers pass KV_CACHE_DTYPE=none (bf16 KV), the setting our numbers use.
#     (24 attn / 4 kv / 16 linear-k / 48 linear-v heads all divide by TP=2.)
#   * Sampling (thinking mode) is carried by the model's shipped
#     generation_config.json (temperature=1.0, top_p=0.95, top_k=20); vLLM applies
#     those as request defaults, so we do NOT override them here — maximally
#     faithful to "what the model released". reasoning_effort=xhigh is the model's
#     own default and is also set explicitly by the agent per request.
#   * chat_template is embedded in tokenizer_config.json (handles thinking /
#     reasoning_effort), so no --chat-template is needed.
#
# Usage:
#   MODEL=/abs/path/to/Qwen3.8-27B SERVED_NAME=qwen38-27b bash serve_qwen38_vllm.sh
# Env knobs (all overridable):
#   DP TP MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL KV_CACHE_DTYPE REASONING_PARSER
#   ENABLE_TOOL_PARSER (1/0) TOOL_CALL_PARSER VLLM_IMAGE PORT
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:?set MODEL to the local Qwen3.8-27B HF dir}"   # absolute path to local HF dir
SERVED_NAME="${SERVED_NAME:-qwen38-27b}"
PORT="${PORT:-8008}"

DP="${DP:-4}"                                   # data-parallel replicas
TP="${TP:-2}"                                   # tensor-parallel per replica (DP*TP = GPUs)
DTYPE="${DTYPE:-bfloat16}"                       # native
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"         # native context
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"             # per-replica concurrent sequences; empty = vLLM default
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"          # recipe: --kv-cache-dtype fp8
REASONING_PARSER="${REASONING_PARSER:-qwen3}"    # recipe: --reasoning-parser qwen3 (splits <think>)
ENABLE_TOOL_PARSER="${ENABLE_TOOL_PARSER:-1}"    # recipe: --enable-auto-tool-choice --tool-call-parser qwen3_coder
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"  # reuse KV of the shared multi-turn history -> big prefill speedup for agentic runs
SPEC_DECODING="${SPEC_DECODING:-1}"                  # MTP speculative decoding (model has mtp_num_hidden_layers=1); recipe-blessed ~1.5-2x decode
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
# LM_ONLY=1 -> --language_model_only: REQUIRED for our RL exports (convert_ckpt_to_hf.sh), which
# carry ONLY the text model + lm_head (no vision tower, no MTP layers) under the base VL config.
# Such exports also need SPEC_DECODING=0 (no mtp.* weights to speculate with). Default 0 = base model.
LM_ONLY="${LM_ONLY:-0}"

# NOTE: MUST satisfy the model's prerequisites — vLLM>=0.17.0 AND transformers>=5.8.0.
# The generic vllm-openai:v0.19.1 ships transformers 5.5.4 (too old → mis-renders the
# thinking/chat-template and tanks scores). The purpose-built qwen38 image bundles
# vLLM(dev) + transformers 5.15.0.
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-x86_64-cu129}"
CONTAINER="${VLLM_CONTAINER:-vllm-${SERVED_NAME}}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
mkdir -p "$HF_CACHE"

[ -d "$MODEL" ] || { echo "ERROR: MODEL dir '$MODEL' not found" >&2; exit 1; }
MOUNT_ARG="-v ${MODEL}:${MODEL}:ro"

TOOL_ARGS=""
[ "$ENABLE_TOOL_PARSER" = "1" ] && TOOL_ARGS="--enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER}"

# KV cache dtype: "none"/"auto"/"bf16"/"" -> OMIT the flag => vLLM keeps the KV cache in the model
# dtype (bf16) = higher fidelity on very long contexts; "fp8" -> quantized KV (faster/smaller).
KV_ARGS=""
case "$KV_CACHE_DTYPE" in
  ""|none|auto|bf16|bfloat16) KV_ARGS="" ;;
  *) KV_ARGS="--kv-cache-dtype ${KV_CACHE_DTYPE}" ;;
esac

SEQS_ARGS=""
[ -n "$MAX_NUM_SEQS" ] && SEQS_ARGS="--max-num-seqs ${MAX_NUM_SEQS}"

PREFIX_ARGS=""
[ "$ENABLE_PREFIX_CACHING" = "1" ] && PREFIX_ARGS="--enable-prefix-caching"

# JSON double-quotes are escaped so they survive expansion inside the double-quoted
# `$DOCKER_SH "..."`; the single quotes around the JSON protect it for the inner shell.
SPEC_ARGS=""
[ "$SPEC_DECODING" = "1" ] && SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_NUM_TOKENS}}'"

LM_ONLY_ARGS=""
[ "$LM_ONLY" = "1" ] && LM_ONLY_ARGS="--language_model_only"

# Fresh container each launch.
$DOCKER_SH "docker rm -f '${CONTAINER}' >/dev/null 2>&1 || true"

# --entrypoint vllm => run the stable `vllm serve <model> [opts]` CLI regardless
# of the image's default entrypoint. --ipc host for vLLM shared memory; --network
# host so harbor (on the host) reaches the API at localhost:${PORT}.
$DOCKER_SH "docker run -d --name '${CONTAINER}' \
  --gpus all --ipc host --network host \
  -e HF_HOME=/hf -v '${HF_CACHE}:/hf' ${MOUNT_ARG} \
  --entrypoint vllm '${IMAGE}' \
  serve '${MODEL}' \
    --served-model-name '${SERVED_NAME}' \
    --data-parallel-size ${DP} --tensor-parallel-size ${TP} \
    --dtype '${DTYPE}' \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    ${SEQS_ARGS} ${KV_ARGS} ${PREFIX_ARGS} ${SPEC_ARGS} ${LM_ONLY_ARGS} \
    --reasoning-parser '${REASONING_PARSER}' \
    ${TOOL_ARGS} \
    --port ${PORT}"

echo "[serve] container '${CONTAINER}' started (image ${IMAGE}) DP=${DP} TP=${TP} len=${MAX_MODEL_LEN} kv=${KV_CACHE_DTYPE:-default} tool_parser=${ENABLE_TOOL_PARSER} prefix_cache=${ENABLE_PREFIX_CACHING} mtp_spec=${SPEC_DECODING} lm_only=${LM_ONLY}."
echo "[serve] logs:  docker logs -f ${CONTAINER}"
echo "[serve] ready: curl -s http://localhost:${PORT}/v1/models"
