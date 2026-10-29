#!/usr/bin/env bash
# =============================================================================
# train_pprs.sh — DPPO RL with progress-potential reward shaping (PPRS) for terminal agents
# (open-instruct grpo_fast.py on Ray); the recipe of the Tev paper.
#
# PPRS = per-turn shaping rewards λ·(γΦₜ−Φₜ₋₁) from the host-side Tev-Gym grader (λ = 0.25, the task
# contract's shaping_scale; γ = 1, so they telescope to λ·Φ(s_T)) plus the strict verifier's terminal reward
# on submit. Credit: per-turn returns-to-go Gₜ (γ=1) with a PER-TURN-INDEX group baseline (each turn's
# advantage is centered by the group mean of Gₜ over the rollouts that reached turn t). Env: `swerl_vanillux_dense`.
#
# Run the SAME command on every node (node 0 = Ray head + trainer; other nodes join Ray and block), with
# the training venv (built from training/open-instruct/pyproject.toml + uv.lock) active so `python` and
# `ray` resolve to it. All paths must be visible at the same location on every node (shared FS or
# identical local copies).
#
# ---- Example: quick start on the shipped one-task example (1 node x 8 GPUs, tiny batches) ----
#   bash scripts/build_sandbox_image.sh                          # builds tev-sandbox:latest (the rows' image)
#   SMOKE=1 MODEL_PATH=/path/to/Qwen3.5-9B bash scripts/train/train_pprs.sh
#   -> DATASET_DIR / V72_CODEBASE default to examples/data/warehouse_etl_example / examples/grader
#      (one task + the grader code for its family).
#
# ---- Example: the 9B recipe on your dataset (4 nodes x 8 GPUs = 16 learners + 16 vLLM engines) ----
#   NUM_NODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node-0 hostname> MODEL_PATH=/path/to/Qwen3.5-9B \
#   DATASET_DIR=/path/to/data/<dataset> V72_CODEBASE=/path/to/grader_codebase \
#   OUTPUT_DIR=/path/to/runs/<exp-name> bash scripts/train/train_pprs.sh
#   -> 8 prompts x 24 rollouts per step; 9216 episodes = 48 steps (the paper's 9B run, on the 264-task
#      unit-tier subset of Tev-Gym).
#
# ---- Example: 27B (Qwen3.8-27B; 8 nodes x 8 A100-80GB = 16 learners + 48 vLLM engines) ----
#   needs a venv with transformers==5.10.0 (Qwen3.8 support; newer releases change the GatedDeltaNet
#   module that open_instruct/qwen3_5_packing_patch.py patches)
#   NUM_NODES=8 VLLM_ENGINES=48 ZPG=1 ASYNC_STEPS=4 RESPONSE_LEN=65536 PACK_LEN=67584 \
#   MODEL_PATH=/path/to/Qwen3.8-27B DATASET_DIR=... V72_CODEBASE=... NODE_RANK=... MASTER_ADDR=... \
#     bash scripts/train/train_pprs.sh
#
# Inputs:
#   MODEL_PATH    HF model dir.
#   DATASET_DIR   dataset dir: DATASET_DIR/data/train-00000-of-00001.parquet (one row per task: prompt messages,
#                 env_config {env_name, image, task_id}) + DATASET_DIR/task-data/<task_id>/ (or task-data.tar.gz,
#                 extracted there on first use): instruction.md, environment/seeds/ (the agent's /app workspace),
#                 authority/package/ (the task's grading package; stays on the host, never enters the sandbox)
#                 and dense_config.json (grade_mode). Each row's env_config.image names the sandbox image
#                 (tev-sandbox:latest from scripts/build_sandbox_image.sh) — every node needs docker access to it.
#   V72_CODEBASE  grader codebase: the directory containing training_data_collection/v72 — it computes the reward
#                 (per-turn progress potential Φ and the terminal score). It must be the exact revision the task
#                 packages were built with (check: scripts/check_grader.py); a mismatched or edited codebase makes
#                 every task silently fall back to terminal-only reward. Never modify examples/grader.
#   MAX_TOOL_CALLS maps to grpo `--max_steps`, which caps TOOL CALLS PER EPISODE (not training steps).
#                 Training length = TOTAL_EPISODES / (UNIQ_PROMPTS x SAMPLES).
#   W&B:          set WANDB_API_KEY (+ WANDB_PROJECT) to enable tracking; restarts of the same EXP_NAME append
#                 to one run. Resume: rerun the same command — grpo resumes from CKPT_DIR (put it on a shared FS
#                 for multi-node runs: every learner rank writes its ZeRO shard there).
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OI="$REPO/training/open-instruct"

# ---- inputs ----
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=/path/to/Qwen3.5-9B}"
DATASET_DIR="${DATASET_DIR:-$REPO/examples/data/warehouse_etl_example}"
V72_CODEBASE="${V72_CODEBASE:-$REPO/examples/grader}"
EXP_NAME="${EXP_NAME:-tev-9b-pprs}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/runs/$EXP_NAME}"
TASK_DATA_DIR="${TASK_DATA_DIR:-$DATASET_DIR/task-data}"
CKPT_DIR="${CKPT_DIR:-$OUTPUT_DIR/checkpoint_state}"
ROLLOUTS_DIR="${ROLLOUTS_DIR:-$OUTPUT_DIR/rollouts}"

# ---- cluster ----
NUM_NODES="${NUM_NODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-8888}"

# ---- recipe (9B values used for the example data; see the 27B example above for its overrides) ----
LEARNERS_PER_NODE="${LEARNERS_PER_NODE:-8 8}"   # per-node learner GPU counts ("8 8" = 16 learners on 2 nodes)
VLLM_ENGINES="${VLLM_ENGINES:-16}"               # learners + VLLM_ENGINES x VLLM_TP must equal NUM_NODES x 8
VLLM_TP="${VLLM_TP:-1}"
SAMPLES="${SAMPLES:-24}"                         # group size (rollouts per prompt)
UNIQ_PROMPTS="${UNIQ_PROMPTS:-8}"                # prompts per step
TOTAL_EPISODES="${TOTAL_EPISODES:-9216}"         # 48 steps at 8 x 24
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-64}"           # per-episode tool-call cap (grpo --max_steps)
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
RESPONSE_LEN="${RESPONSE_LEN:-65536}"
PACK_LEN="${PACK_LEN:-67584}"                    # >= MAX_PROMPT_LEN + RESPONSE_LEN
PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:-16384}"
ASYNC_STEPS="${ASYNC_STEPS:-6}"
SEQ_PARALLEL="${SEQ_PARALLEL:-4}"
ZPG="${ZPG:-8}"                                  # ZeRO++ hierarchical partitioning (8 works with 16 learners; 1 = off)
POOL_SIZE="${POOL_SIZE:-768}"                    # sandbox pool; smaller pools stall rollouts in env reset
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.9}"
LR="${LR:-1e-6}"
CKPT_FREQ="${CKPT_FREQ:-2}"
CKPT_KEEP="${CKPT_KEEP:-2}"
SAVE_FREQ="${SAVE_FREQ:-20}"
SHAPING_GAMMA="${SHAPING_GAMMA:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VLLM_FLAGS="--vllm_gpu_memory_utilization $VLLM_GPU_MEM"

if [ "${SMOKE:-0}" = "1" ]; then
  NUM_NODES=1; LEARNERS_PER_NODE="4"; VLLM_ENGINES=4; SAMPLES=8; UNIQ_PROMPTS=2; TOTAL_EPISODES=32
  MAX_TOOL_CALLS=24; RESPONSE_LEN=8192; PACK_LEN=10240; PER_TURN_MAX_TOKENS=4096; ASYNC_STEPS=2; SEQ_PARALLEL=1
  ZPG=1; POOL_SIZE=16; CKPT_FREQ=1000000; SAVE_FREQ=1000000
  VLLM_FLAGS="--vllm_gpu_memory_utilization 0.6 --vllm_enforce_eager --vllm_sync_backend gloo"
fi

# ---- preflight ----
[ -f "$MODEL_PATH/config.json" ] || { echo "ERROR: no config.json in MODEL_PATH=$MODEL_PATH" >&2; exit 1; }
# The learner->vLLM weight sync maps HF names to vLLM's `language_model.*` names only when the model path
# names a Qwen3.5/3.6/3.8 model (grpo_fast._build_vlm_name_mapper), so keep that in the path.
case "$(printf '%s' "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')" in
  *qwen3.5*|*qwen3.6*|*qwen3.8*) ;;
  *) echo "WARNING: MODEL_PATH=$MODEL_PATH does not contain qwen3.5/qwen3.6/qwen3.8; for these models the" \
          "vLLM weight-name mapping is then skipped. Use a path like /path/to/Qwen3.5-9B-<name>." >&2 ;;
esac
[ -f "$DATASET_DIR/data/train-00000-of-00001.parquet" ] || { echo "ERROR: $DATASET_DIR/data/train-00000-of-00001.parquet missing" >&2; exit 1; }
[ -d "$V72_CODEBASE/training_data_collection/v72" ] || { echo "ERROR: $V72_CODEBASE/training_data_collection/v72 missing" >&2; exit 1; }
LGPU=0; for n in $LEARNERS_PER_NODE; do LGPU=$((LGPU + n)); done
[ $((LGPU + VLLM_ENGINES * VLLM_TP)) -eq $((NUM_NODES * 8)) ] || \
  echo "WARNING: learners ($LGPU) + vLLM GPUs ($((VLLM_ENGINES * VLLM_TP))) != NUM_NODES x 8 ($((NUM_NODES * 8)))" >&2
[ "$PACK_LEN" -ge $((MAX_PROMPT_LEN + RESPONSE_LEN)) ] || { echo "ERROR: PACK_LEN < MAX_PROMPT_LEN + RESPONSE_LEN" >&2; exit 1; }
# task-data: use an existing extracted dir, else extract task-data.tar.gz once (flock + marker, so nodes that
# share a filesystem neither race nor start on a half-extracted tree).
if [ ! -f "$TASK_DATA_DIR/.extracted" ] && [ -z "$(ls -A "$TASK_DATA_DIR" 2>/dev/null)" ]; then
  [ -f "$DATASET_DIR/task-data.tar.gz" ] || { echo "ERROR: neither $TASK_DATA_DIR nor $DATASET_DIR/task-data.tar.gz" >&2; exit 1; }
  mkdir -p "$TASK_DATA_DIR"
  (
    flock 9
    if [ ! -f "$TASK_DATA_DIR/.extracted" ]; then
      echo "[train_pprs] extracting task-data.tar.gz -> $TASK_DATA_DIR"
      tar -C "$TASK_DATA_DIR" -xzf "$DATASET_DIR/task-data.tar.gz" && touch "$TASK_DATA_DIR/.extracted"
    fi
  ) 9>"$TASK_DATA_DIR.lock"
  [ -f "$TASK_DATA_DIR/.extracted" ] || { echo "ERROR: task-data extraction failed" >&2; exit 1; }
fi
# git cannot store empty dirs: recreate each task's (empty) environment/seeds/output, i.e. the agent's /app/output.
find "$TASK_DATA_DIR" -mindepth 3 -maxdepth 3 -type d -path '*/environment/seeds' -exec mkdir -p {}/output \; 2>/dev/null || true
# prompts per step cannot exceed the number of tasks (the loader drops incomplete batches) — e.g. the one-task example.
N_TASKS="$(python -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" \
  "$DATASET_DIR/data/train-00000-of-00001.parquet" 2>/dev/null || echo "")"
if [ -n "$N_TASKS" ] && [ "$N_TASKS" -lt "$UNIQ_PROMPTS" ]; then
  echo "WARNING: dataset has $N_TASKS task(s) < UNIQ_PROMPTS=$UNIQ_PROMPTS; using UNIQ_PROMPTS=$N_TASKS" >&2
  UNIQ_PROMPTS="$N_TASKS"
fi
docker info >/dev/null 2>&1 || echo "WARNING: docker is not reachable by $(id -un); the sandbox env needs it on every node" >&2

# ---- runtime env ----
export PYTHONPATH="$OI${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 NCCL_CUMEM_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SWERL_RESET_FAILURE_ZERO_REWARD=1 SWERL_DOCKER_AUTO_REMOVE=1 SWERL_SANDBOX_TIMING_LOGS=1
export SWERL_POOL_INIT_MAX_ATTEMPTS="${SWERL_POOL_INIT_MAX_ATTEMPTS:-3}" SWERL_POOL_INIT_RETRY_BACKOFF_S="${SWERL_POOL_INIT_RETRY_BACKOFF_S:-5}"
ulimit -n 1048576 2>/dev/null || ulimit -n 262144 2>/dev/null || true   # vLLM engines + sandbox pool open many fds

# Blackwell (compute capability >= 10): pin the learner to flash-attention-2 (Ulysses sequence parallelism
# rejects the auto-selected flash-attention-4).
GPU_CC_MAJOR="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
if [ -n "${ATTN_IMPL:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --attn_implementation $ATTN_IMPL"
elif [ "${GPU_CC_MAJOR:-0}" -ge 10 ] 2>/dev/null; then EXTRA_ARGS="$EXTRA_ARGS --attn_implementation flash_2"; fi

TRACKING=""
if [ -n "${WANDB_API_KEY:-}" ]; then
  TRACKING="--with_tracking --wandb_project ${WANDB_PROJECT:-tev-pprs}"
  export WANDB_RUN_ID="${WANDB_RUN_ID:-$EXP_NAME}" WANDB_RESUME="${WANDB_RESUME:-allow}"
fi

TOOL_CONFIGS="$(TASK_DATA_DIR="$TASK_DATA_DIR" V72_CODEBASE="$V72_CODEBASE" SHAPING_GAMMA="$SHAPING_GAMMA" python -c '
import json, os
print(json.dumps({
    "task_data_dir": os.environ["TASK_DATA_DIR"],
    "test_timeout": 120,
    "grade_mode": "live",          # default for a task without dense_config.json (each Tev-Gym task ships one)
    "require_grader": True,        # fail loudly if the grader (V72_CODEBASE) cannot be imported
    "shaping_gamma": float(os.environ["SHAPING_GAMMA"]),
    "v72_codebase_path": os.environ["V72_CODEBASE"],
}))')"

# ---- Ray: node 0 = head, others join and block until the head goes away ----
ray stop --force >/dev/null 2>&1 || true
if [ "$NODE_RANK" != "0" ]; then
  LEADER_IP="$(getent hosts "$MASTER_ADDR" | awk '{print $1}')"; LEADER_IP="${LEADER_IP:-$MASTER_ADDR}"
  export RAY_ADDRESS="$LEADER_IP:$RAY_PORT"
  for _ in $(seq 1 180); do ray start --address="$RAY_ADDRESS" && break; echo "[train_pprs] waiting for Ray head at $RAY_ADDRESS"; sleep 5; done
  ray status --address="$RAY_ADDRESS" >/dev/null 2>&1 || { echo "ERROR: could not join Ray head $RAY_ADDRESS" >&2; exit 1; }
  echo "[train_pprs] node $NODE_RANK joined $RAY_ADDRESS; blocking until the head exits"
  misses=0
  while [ "$misses" -lt 60 ]; do
    if ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then misses=0; else misses=$((misses + 1)); fi
    sleep 5
  done
  ray stop --force >/dev/null 2>&1 || true
  exit 0
fi
ray start --head --port="$RAY_PORT" --dashboard-host=0.0.0.0
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
if [ "$NUM_NODES" -gt 1 ]; then
  echo "[train_pprs] waiting for $NUM_NODES nodes to join Ray ..."
  for _ in $(seq 1 360); do
    n="$(python -c 'import ray; ray.init(address="auto", logging_level="ERROR"); print(sum(1 for x in ray.nodes() if x["Alive"]))' 2>/dev/null || echo 0)"
    [ "$n" -ge "$NUM_NODES" ] && break; sleep 5
  done
fi

# ---- train (node 0) ----
mkdir -p "$OUTPUT_DIR" "$ROLLOUTS_DIR"
# grpo resumes from CKPT_DIR when it holds a checkpoint; an EMPTY dir makes it try (and fail) to load one.
[ -d "$CKPT_DIR" ] && [ -z "$(ls -A "$CKPT_DIR")" ] && rmdir "$CKPT_DIR"
cd "$OI"
set +e
python -u open_instruct/grpo_fast.py \
    --dataset_mixer_list "$DATASET_DIR" 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length "$MAX_PROMPT_LEN" \
    --per_turn_max_tokens "$PER_TURN_MAX_TOKENS" \
    --response_length "$RESPONSE_LEN" \
    --pack_length "$PACK_LEN" \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "$UNIQ_PROMPTS" \
    --num_samples_per_prompt_rollout "$SAMPLES" \
    --async_steps "$ASYNC_STEPS" \
    --model_name_or_path "$MODEL_PATH" \
    --temperature 1.0 \
    --learning_rate "$LR" \
    --total_episodes "$TOTAL_EPISODES" \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --deepspeed_zpg "$ZPG" \
    --gather_whole_model false \
    --sequence_parallel_size "$SEQ_PARALLEL" \
    --num_epochs 1 \
    --num_learners_per_node $LEARNERS_PER_NODE \
    --vllm_num_engines "$VLLM_ENGINES" \
    --vllm_tensor_parallel_size "$VLLM_TP" \
    $VLLM_FLAGS \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --save_traces \
    --tools swerl_vanillux_dense \
    --tool_configs "$TOOL_CONFIGS" \
    --pool_size "$POOL_SIZE" \
    --active_sampling \
    --max_steps "$MAX_TOOL_CALLS" \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/prompts/swerl_vanillux_sandbox_system_prompt.txt \
    --filter_zero_std_samples true \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --inflight_updates true \
    --lm_head_fp32 true --use_liger_grpo_loss --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1 \
    --reward_aggregator last --per_turn_credit --per_turn_credit_gamma 1.0 \
    --rollouts_save_path "$ROLLOUTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --local_eval_every 1000000 \
    --checkpoint_state_dir "$CKPT_DIR" \
    --checkpoint_state_freq "$CKPT_FREQ" \
    --keep_last_n_checkpoints "$CKPT_KEEP" \
    --save_freq "$SAVE_FREQ" \
    --save_trainer_logprobs true \
    $TRACKING $EXTRA_ARGS 2>&1 | tee -a "$OUTPUT_DIR/train.log"
RC=${PIPESTATUS[0]}
set -e
echo "[train_pprs] TRAIN_EXIT=$RC"
ray stop --force >/dev/null 2>&1 || true
exit "$RC"
