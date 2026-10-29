# open-instruct (trimmed fork)

A trimmed fork of [allenai/open-instruct](https://github.com/allenai/open-instruct), reduced to
what `open_instruct/grpo_fast.py` needs for the Tev PPRS (progress-potential reward shaping) DPPO runs:
- the RL trainer, data loader, vLLM rollout actors and utils;
- the sandbox envs (`open_instruct/environments/`);
- the unit tests for those modules.

SFT, DPO, reward modeling, OLMo-core training, distillation and the upstream
scripts/configs/docs were removed.

Fork changes on top of upstream that the Tev runs rely on:
- **Qwen 3.5 / 3.8** support: packed-sequence GatedDeltaNet patch in `open_instruct/qwen3_5_packing_patch.py`.
- **DPPO** loss (`--loss_fn dppo`).
- **Terminal-agent sandboxes**: `swerl_vanillux_sandbox`, and the dense `swerl_vanillux_dense` env
  with its host-side Tev-Gym grader driver (`environments/v72_grader.py`).
- **PPRS per-turn credit** (`--per_turn_credit`, `open_instruct/dense_credit.py`): discounted
  per-turn returns with a per-timestep group baseline, scattered onto each turn's tokens.

It is launched by `scripts/train/train_pprs.sh` (see the repo-root `README.md`) inside a venv built
from this `pyproject.toml` / `uv.lock` (`uv sync`).

Tests (CPU): `PYTHONPATH=. python -m pytest open_instruct tests --ignore-glob='*_gpu.py'`.
