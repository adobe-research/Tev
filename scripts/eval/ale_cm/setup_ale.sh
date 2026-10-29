#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_ale.sh — prepare an Agents' Last Exam (ALE) checkout for the computing_math eval:
#   1. clone ALE (https://github.com/rdi-berkeley/agents-last-exam) at ALE_REF, unless ALE_DIR exists
#   2. install this repo's vanillux_oai agent into it (ale_run/agents/vanillux_oai)
#   3. build its venv (uv; framework + every agent + task-grader deps, incl. torch)
#   4. fetch the task data (gated Hugging Face dataset; needs approved access + `hf auth login`)
#   5. pull the sandbox image agentslastexam/ale-ubuntu22-docker:latest
#
# ALE_REF defaults to the upstream revision our reported numbers were produced with. Later upstream
# revisions changed the prompts/graders of some computing_math tasks, so scores may differ there.
# An existing ALE_DIR is used as-is (no checkout); only the agent package is (re)installed into it.
#
# Idempotent: re-run cheaply; finished steps are skipped.
#
# Usage:
#   bash scripts/eval/ale_cm/setup_ale.sh
#   ALE_DIR=/path/to/agents-last-exam TASK_DATA=/path/to/task-data bash scripts/eval/ale_cm/setup_ale.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK="${WORK:-$HOME/.cache/tev_eval}"
ALE_REPO_URL="${ALE_REPO_URL:-https://github.com/rdi-berkeley/agents-last-exam.git}"
ALE_REF="${ALE_REF:-43c3d8e931220b03fcebe50ad806e2ca27690c06}"
ALE_DIR="${ALE_DIR:-$WORK/agents-last-exam}"
TASK_DATA="${TASK_DATA:-$ALE_DIR/task-data}"       # <root>/<domain>/<task>/<variant>/{input,software,reference}
ALE_IMAGE="${ALE_IMAGE:-agentslastexam/ale-ubuntu22-docker:latest}"
HF_DATA_REPO="${HF_DATA_REPO:-agents-last-exam/agents-last-exam-data-archive}"
HF_DATA_FILE="${HF_DATA_FILE:-ale-tasks-data.tar.gz}"
DOCKER_SH="${DOCKER_SH:-bash -c}"   # set DOCKER_SH='sg docker -c' if your docker group membership is stale

say() { echo "[setup_ale $(date -u +%FT%TZ)] $*"; }
die() { say "FATAL: $*"; exit 1; }
command -v uv >/dev/null 2>&1 || die "uv not found (https://docs.astral.sh/uv/)"
command -v git >/dev/null 2>&1 || die "git not found"
mkdir -p "$WORK"

# 1. ALE checkout
if [ ! -d "$ALE_DIR" ]; then
  say "cloning $ALE_REPO_URL -> $ALE_DIR @ $ALE_REF"
  git clone "$ALE_REPO_URL" "$ALE_DIR"
  git -C "$ALE_DIR" checkout -q "$ALE_REF"
else
  head="$(git -C "$ALE_DIR" rev-parse HEAD 2>/dev/null || echo "<not a git checkout>")"
  [ "$head" = "$(git -C "$ALE_DIR" rev-parse -q --verify "$ALE_REF^{commit}" 2>/dev/null || true)" ] \
    || say "WARN: $ALE_DIR is at $head, not ALE_REF=$ALE_REF (using it as-is)"
fi
[ -f "$ALE_DIR/pyproject.toml" ] && [ -d "$ALE_DIR/ale_run" ] || die "$ALE_DIR does not look like an ALE checkout"

# 2. agent package (a uv workspace member: ale_run/agents/*)
rm -rf "$ALE_DIR/ale_run/agents/vanillux_oai"
mkdir -p "$ALE_DIR/ale_run/agents/vanillux_oai"
cp "$HERE"/vanillux_oai/{__init__.py,config.py,deployer.py,pyproject.toml} "$ALE_DIR/ale_run/agents/vanillux_oai/"
say "installed vanillux_oai -> $ALE_DIR/ale_run/agents/vanillux_oai"

# 3. venv. --all-packages pulls the task-grader deps (the graders run on the host; some import torch).
#    cua-bench is built from source: its published wheel is malformed.
cd "$ALE_DIR"
if ! .venv/bin/python -c "import torch, cua_bench; from ale_run.agents.vanillux_oai.deployer import VanilluxOaiDeployer" 2>/dev/null; then
  say "uv sync (framework + agents + task-grader deps) ..."
  uv sync --all-packages --extra dev --no-binary-package cua-bench
fi
.venv/bin/python -c "import torch, cua_bench; from ale_run.agents.vanillux_oai.deployer import VanilluxOaiDeployer" \
  || die "ALE venv import check failed ($ALE_DIR/.venv)"
say "venv ok: $ALE_DIR/.venv"

# 4. task data
if [ ! -d "$TASK_DATA/computing_math" ]; then
  say "downloading $HF_DATA_FILE from $HF_DATA_REPO (gated: request access on Hugging Face, then 'hf auth login')"
  mkdir -p "$TASK_DATA"
  .venv/bin/python - "$HF_DATA_REPO" "$HF_DATA_FILE" "$TASK_DATA" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, fname, dest = sys.argv[1:]
hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=dest)
PY
  tar xzf "$TASK_DATA/$HF_DATA_FILE" -C "$TASK_DATA"
  rm -f "$TASK_DATA/$HF_DATA_FILE"
fi
missing=0
while read -r t; do
  [ -d "$TASK_DATA/$t" ] || { say "WARN: no task data for $t under $TASK_DATA"; missing=1; }
done < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$HERE/computing_math.txt")
[ "$missing" = 0 ] || die "task data incomplete under $TASK_DATA"
say "task data ok: $TASK_DATA"

# 5. sandbox image
$DOCKER_SH "docker image inspect '$ALE_IMAGE' >/dev/null 2>&1 || docker pull '$ALE_IMAGE'" || die "docker pull $ALE_IMAGE"
say "image ok: $ALE_IMAGE"
say "DONE (ALE_DIR=$ALE_DIR)"
