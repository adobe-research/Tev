#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# convert_ckpt_to_hf.sh — convert ONE DeepSpeed ZeRO checkpoint written by train_pprs.sh
# (CKPT_DIR/global_step$STEP) into a servable HF model dir, for the Qwen3.5-9B / Qwen3.8-27B
# (model_type qwen3_5, VL wrapper) models:
#   get_fp32_state_dict_from_zero_checkpoint(tag) -> drop lm_head only if tied to embed_tokens
#   -> remap `model.*` -> `model.language_model.*` (skip already-prefixed / model.visual.*)
#   -> save ONE model.safetensors (DTYPE fp32 or bf16) + the base model's config/tokenizer files
#      (NOT its weight index).
# The RL checkpoint holds ONLY the text model (+lm_head); vision / MTP weights stay in the base model,
# so serve the export with --language_model_only (LM_ONLY=1, the eval_tb21.sh default) and, for Qwen3.8,
# without MTP speculative decoding (SPEC_DECODING=0).
#
# Usage:
#   CKPT_DIR=/path/to/runs/<exp-name>/checkpoint_state STEP=48 \
#   BASE_MODEL_DIR=/path/to/Qwen3.5-9B OUT_DIR=/path/to/hf/<exp-name>-step48 \
#     bash scripts/eval/convert_ckpt_to_hf.sh
#   (27B: BASE_MODEL_DIR=/path/to/Qwen3.8-27B DTYPE=bf16)
# Needs a python with torch + deepspeed + safetensors (e.g. the training venv): PYTHON=/path/to/python.
# Reconstruction happens in CPU RAM in fp32 (~4 bytes/param: ~36 GB for 9B, ~110 GB for 27B).
# ---------------------------------------------------------------------------
set -euo pipefail
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR=<checkpoint_state dir of the run, containing global_step<N>/>}"
STEP="${STEP:?set STEP=<checkpoint step, e.g. 48>}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:?set BASE_MODEL_DIR=/path/to/<base HF model> (config/tokenizer source)}"
OUT_DIR="${OUT_DIR:?set OUT_DIR=/path/to/hf/<name>}"
DTYPE="${DTYPE:-fp32}"            # fp32 (our 9B exports) | bf16 (our 27B exports)
PYTHON="${PYTHON:-python}"
TAG="global_step$STEP"
[ -d "$CKPT_DIR/$TAG" ] || { echo "ERROR: $CKPT_DIR/$TAG not found" >&2; exit 1; }
[ -f "$BASE_MODEL_DIR/config.json" ] || { echo "ERROR: no config.json in $BASE_MODEL_DIR" >&2; exit 1; }
case "$DTYPE" in fp32|bf16) ;; *) echo "ERROR: DTYPE must be fp32 or bf16" >&2; exit 1 ;; esac
say(){ echo "[convert $TAG $(date +%H:%M:%S)] $*"; }

mkdir -p "$OUT_DIR"
say "reconstructing $CKPT_DIR/$TAG -> $OUT_DIR/model.safetensors ($DTYPE)"
CKDIR="$CKPT_DIR" TAG="$TAG" OUTW="$OUT_DIR" DTYPE="$DTYPE" "$PYTHON" - <<'PYEOF'
import os, torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import save_file
ckdir, tag, out, dtype = os.environ["CKDIR"], os.environ["TAG"], os.environ["OUTW"], os.environ["DTYPE"]
sd = get_fp32_state_dict_from_zero_checkpoint(ckdir, tag=tag)
print(f"reconstructed {len(sd)} tensors from {tag}")
lm, emb = sd.get("lm_head.weight"), sd.get("model.embed_tokens.weight")
if lm is not None and emb is not None and lm.data_ptr() == emb.data_ptr():
    del sd["lm_head.weight"]; print("dropped TIED lm_head.weight")
else:
    print("lm_head kept (untied)" if lm is not None else "no lm_head in ckpt")
def remap(k):  # text model model.* -> model.language_model.* (qwen3_5 VL wrapper); guard against double-prefix
    if k.startswith("model.") and not k.startswith("model.language_model.") and not k.startswith("model.visual."):
        return "model.language_model." + k[len("model."):]
    return k
cast = (lambda v: v.to(torch.bfloat16)) if dtype == "bf16" else (lambda v: v)
sd = {remap(k): cast(v).contiguous() for k, v in sd.items()}
save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
tot = sum(v.numel() for v in sd.values())
print(f"saved {len(sd)} tensors, {tot} elems ({dtype}) -> {out}")
PYEOF

# config/tokenizer from the base model (NOT model.safetensors.index.json: the export is a single file)
for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja \
         special_tokens_map.json vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$BASE_MODEL_DIR/$f" ] && cp "$BASE_MODEL_DIR/$f" "$OUT_DIR/"
done

# verify: every exported key exists in the base model's weight index (i.e. the remap is right)
"$PYTHON" - "$OUT_DIR" "$BASE_MODEL_DIR/model.safetensors.index.json" <<'PYEOF'
import json, os, sys
from safetensors import safe_open
out, idx = sys.argv[1], sys.argv[2]
with safe_open(os.path.join(out, "model.safetensors"), framework="pt") as f:
    keys = set(f.keys())
if not os.path.exists(idx):
    print(f"[verify] {len(keys)} keys (no base index at {idx}; skipped key check)"); sys.exit(0)
base = set(json.load(open(idx))["weight_map"])
bad = sorted(k for k in keys if k not in base)
print(f"[verify] {len(keys)} keys; not in base index: {len(bad)} {bad[:3]}")
sys.exit(1 if bad else 0)
PYEOF
say "DONE: $OUT_DIR"; ls -la "$OUT_DIR" | sed 's/^/    /'
