"""Per-turn (dense-reward) credit assignment for GRPO-family RL — the PPRS (``return_pt``) scheme.

It replaces the default
"one group-relative scalar per rollout broadcast over every response token" with **per-action returns
assigned only to the token span responsible for that action**.

Pure logic, no vLLM / Ray / trainer imports so it is unit-testable in isolation. The trainer wires
it in `data_loader.py` behind the `--per_turn_credit` flag; when the flag is off nothing here runs
and behaviour is byte-identical to the scalar-broadcast path.

Design (critic-free):
  * Each rollout i produces a per-turn immediate reward vector `turn_rewards_i` (the env's
    potential-shaping deltas λ·(γΦₜ−Φₜ₋₁), already in `rollout_states[i]["rewards"]`) with the terminal
    reward (the reward-contract score) in the LAST slot.
  * Discounted returns run backward with the training gamma (matches v72
    `contracts/progress.py:build_action_reward_assignments`, kept as the test oracle):
        G_t = immediate_t + gamma * G_{t+1}
  * PER-TIMESTEP group baseline: at each turn index t, subtract the group mean of G_t over the rollouts
    of the same prompt group that reached turn t (the active set I_t). This makes the advantage
    zero-mean at EVERY turn — a single trajectory-level G_0 baseline would leave a systematic
    later-turn negative offset (mean_j Ã_t = -λ·Φ̄_{t-1}) that biases the policy toward terser,
    submit-sooner behaviour — while preserving cross-turn credit propagation (γ=1) and the
    potential-based shaping.
  * The per-turn advantage is then scattered onto that turn's assistant token span.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_per_turn_returns",
    "build_token_advantages",
    "derive_turn_ids_from_mask",
    "compute_per_turn_advantages_pt",
    "build_rollout_token_advantages_return_pt",
]


def compute_per_turn_returns(
    turn_rewards: list[float],
    terminal_reward: float,
    gamma: float,
) -> list[float]:
    """Backward discounted returns over one rollout's action-level rewards.

    `turn_rewards[t]` is the immediate shaping reward attributed to action t (0 for read-only /
    no-change turns). `terminal_reward` is the reward-contract score, added to the final action
    only (matching v72 `build_action_reward_assignments`: `immediate[-1] += terminal_reward`).

    Returns a list `G` with `G[t] = immediate[t] + gamma * G[t+1]` (len == len(turn_rewards)).
    If `turn_rewards` is empty, the terminal is treated as a single-action rollout.
    """
    immediate = [float(r) for r in turn_rewards] if turn_rewards else [0.0]
    immediate[-1] += float(terminal_reward)
    returns = [0.0] * len(immediate)
    running = 0.0
    for t in range(len(immediate) - 1, -1, -1):
        running = immediate[t] + gamma * running
        returns[t] = running
    return returns


def build_token_advantages(
    per_turn_advantages: list[list[float]],
    turn_ids: list[int],
    rollout_id: int,
) -> list[float]:
    """Map a rollout's per-turn advantages onto its per-token turn-id stream.

    `turn_ids[k]` is the 1-based action index that produced response token k (0 = query / tool /
    non-credited token). `rollout_id` selects `per_turn_advantages[rollout_id]`. A `turn_ids` value
    of `j` (>=1) maps to `per_turn_advantages[rollout_id][j-1]`; 0 maps to advantage 0.0.

    Returns a per-token advantage list aligned 1:1 with `turn_ids`.
    """
    adv = per_turn_advantages[rollout_id]
    out = []
    for tid in turn_ids:
        if tid <= 0 or tid > len(adv):
            out.append(0.0)
        else:
            out.append(adv[tid - 1])
    return out


def derive_turn_ids_from_mask(mask: list[int]) -> list[int]:
    """Turn (action) index per response token, derived from the binary assistant/tool mask.

    `mask[k] == 1` for assistant-generated tokens, 0 for tool-output tokens. A maximal run of 1s is
    one turn; the turn counter increments at each 0→1 transition (1-based). Tool tokens get 0 (no
    credit — they are masked from the policy loss anyway).
    """
    turn_ids: list[int] = []
    cur_turn = 0
    prev = 0
    for m in mask:
        if m:
            if not prev:
                cur_turn += 1
            turn_ids.append(cur_turn)
        else:
            turn_ids.append(0)
        prev = m
    return turn_ids


def compute_per_turn_advantages_pt(
    per_rollout_returns: list[list[float]],
    num_samples_per_prompt: int,
    advantage_normalization_type: str = "centered",
    eps: float = 1e-8,
) -> list[list[float]]:
    """Group-relative per-turn advantages with a PER-TIMESTEP baseline.

    `per_rollout_returns[i]` is rollout i's per-turn discounted returns (from `compute_per_turn_returns`).
    Rollouts are grouped in consecutive blocks of `num_samples_per_prompt` (the GRPO group, same
    convention as `compute_group_advantages`). For each group and each turn index t the baseline is the
    group mean of G_t over the rollouts that reached turn t — the active set
    I_{g,t} = {i in g : t < len(returns_i)}. For `advantage_normalization_type == "standard"` the
    per-timestep group std additionally divides; `"centered"` (default) only subtracts the per-timestep
    mean. Returns per-rollout per-turn advantage lists, same ragged shape as the input. A turn with a
    single active rollout centers to 0 (and, on the standard path, the +eps denom keeps it finite)."""
    n = len(per_rollout_returns)
    if n == 0:
        return []
    if n % num_samples_per_prompt != 0:
        raise ValueError(
            f"per_rollout_returns length {n} not divisible by num_samples_per_prompt {num_samples_per_prompt}"
        )
    if advantage_normalization_type not in ("centered", "standard"):
        raise ValueError(f"unsupported advantage_normalization_type: {advantage_normalization_type!r}")
    out: list[list[float]] = [[0.0] * len(r) for r in per_rollout_returns]
    for g0 in range(0, n, num_samples_per_prompt):
        group = range(g0, g0 + num_samples_per_prompt)
        max_turns = max((len(per_rollout_returns[i]) for i in group), default=0)
        for t in range(max_turns):
            active = [i for i in group if t < len(per_rollout_returns[i])]
            if not active:
                continue
            vals = np.array([per_rollout_returns[i][t] for i in active], dtype=np.float64)
            mu = float(vals.mean())
            denom = (float(vals.std()) + eps) if advantage_normalization_type == "standard" else 1.0
            for i in active:
                out[i][t] = float((per_rollout_returns[i][t] - mu) / denom)
    return out


def build_rollout_token_advantages_return_pt(
    masks: list[list[int]],
    per_rollout_rewards: list[list[float]],
    num_samples_per_prompt: int,
    gamma: float,
    advantage_normalization_type: str = "centered",
    eps: float = 1e-8,
) -> list[list[float]]:
    """Per-token advantages for every rollout (aligned to each rollout's response tokens).

    `masks[i]` is rollout i's binary assistant/tool mask (len == len(response_i)). `per_rollout_rewards[i]`
    is its per-turn immediate reward vector (env shaping deltas with the terminal reward already in the
    last slot — so `terminal_reward=0` here). Computes discounted returns G_t with discount `gamma`, the
    per-timestep baseline (`compute_per_turn_advantages_pt`), then scatters each turn's advantage onto
    that turn's assistant token span. Returns per-rollout per-token advantage lists, same ragged shape as
    `masks`, ready to thread through `pack_sequences(per_token_advantages=...)`."""
    if len(masks) != len(per_rollout_rewards):
        raise ValueError(f"masks ({len(masks)}) and per_rollout_rewards ({len(per_rollout_rewards)}) length mismatch")
    per_rollout_returns = [
        compute_per_turn_returns(rewards, terminal_reward=0.0, gamma=gamma) for rewards in per_rollout_rewards
    ]
    per_turn_adv = compute_per_turn_advantages_pt(
        per_rollout_returns, num_samples_per_prompt, advantage_normalization_type, eps
    )
    out: list[list[float]] = []
    for i, mask in enumerate(masks):
        turn_ids = derive_turn_ids_from_mask(mask)
        out.append(build_token_advantages(per_turn_adv, turn_ids, rollout_id=i))
    return out
