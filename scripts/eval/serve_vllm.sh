#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# serve_vllm.sh — serve a Qwen3.5-family model (a Tev-9B checkpoint or the base) with the official vLLM OpenAI docker
# image on localhost:$PORT, data-parallel across the node's GPUs.
#
# Uses the prebuilt vLLM OpenAI docker image (torch + CUDA + vLLM included), so there is no wheel install.
#
# Usage:
#   MODEL=/abs/path/to/hf_model SERVED_NAME=tev-9b bash serve_vllm.sh
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:?set MODEL=<HF id or absolute path to a local HF dir>}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
PORT="${PORT:-8008}"
DP="${DP:-8}"                              # data-parallel replicas (a 9B fits 1 A100-80GB)
TP="${TP:-1}"                              # tensor-parallel (1 for a 9B model)
DTYPE="${DTYPE:-auto}"                     # 'auto' = model native; 'bfloat16' to force
PARSER="${PARSER:-qwen3_xml}"              # the tool-call parser the Qwen3.5 model card recommends
# REASONING_PARSER: split <think> reasoning from content/tool-calls. EMPTY by default (the setting our
# 9B numbers use with vLLM v0.19.1). On newer vLLM images a thinking model may need it so <think>
# content doesn't leak into the tool-call parse (serve_qwen38_vllm.sh uses --reasoning-parser qwen3).
# Set REASONING_PARSER=qwen3 to add it.
REASONING_PARSER="${REASONING_PARSER:-}"
LM_ONLY="${LM_ONLY:-1}"                    # our Qwen3.5 checkpoints are text-only on a multimodal-capable
                                           # base (no preprocessor_config.json) -> vLLM needs
                                           # --language_model_only or it fails loading an image
                                           # processor
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.1}"
CONTAINER="${VLLM_CONTAINER:-vllm-${SERVED_NAME}}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale
mkdir -p "$HF_CACHE"

# If MODEL is a local directory, bind-mount it into the container at the same path.
MOUNT_ARG=""
if [ -d "$MODEL" ]; then MOUNT_ARG="-v ${MODEL}:${MODEL}:ro"; fi

LM_ONLY_ARG=""
[ "$LM_ONLY" = "1" ] && LM_ONLY_ARG="--language_model_only"
REASONING_ARG=""
[ -n "$REASONING_PARSER" ] && REASONING_ARG="--reasoning-parser ${REASONING_PARSER}"

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
    --enable-auto-tool-choice --tool-call-parser '${PARSER}' ${REASONING_ARG} \
    --data-parallel-size ${DP} --tensor-parallel-size ${TP} \
    --dtype '${DTYPE}' ${LM_ONLY_ARG} --port ${PORT}"

echo "[serve] container '${CONTAINER}' started (image ${IMAGE})."
echo "[serve] logs:  docker logs -f ${CONTAINER}"
echo "[serve] ready: curl -s http://localhost:${PORT}/v1/models"
