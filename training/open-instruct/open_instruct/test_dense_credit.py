"""Unit tests for per-turn dense-reward credit assignment (open_instruct/dense_credit.py).

The returns math is validated against the v72 reference implementation
(`training_data_collection.v72.contracts.progress.build_action_reward_assignments`) when the
grader codebase is importable ($V72_CODEBASE), and against a hand-computed expectation otherwise.

Run: uv run pytest open_instruct/test_dense_credit.py  (or: python -m pytest ...)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from open_instruct.dense_credit import (
    build_rollout_token_advantages_return_pt,
    build_token_advantages,
    compute_per_turn_advantages_pt,
    compute_per_turn_returns,
    derive_turn_ids_from_mask,
)

# Tev-Gym grader codebase (the reference oracle; e.g. examples/grader), used only if $V72_CODEBASE points at it.
_V72_CODEBASE = Path(os.environ.get("V72_CODEBASE", "/nonexistent"))
_HAVE_V72 = (_V72_CODEBASE / "training_data_collection" / "v72" / "contracts" / "progress.py").is_file()


def test_returns_backward_pass_basic():
    # immediate = [0.1, 0.2, 0.0], terminal 1.0 -> immediate' = [0.1, 0.2, 1.0]
    g = compute_per_turn_returns([0.1, 0.2, 0.0], terminal_reward=1.0, gamma=1.0)
    assert g == pytest.approx([1.3, 1.2, 1.0])


def test_returns_discounted():
    g = compute_per_turn_returns([0.0, 0.0], terminal_reward=1.0, gamma=0.9)
    # immediate' = [0.0, 1.0]; G1 = 1.0; G0 = 0 + 0.9*1.0 = 0.9
    assert g == pytest.approx([0.9, 1.0])


def test_returns_empty_turns_is_single_action():
    g = compute_per_turn_returns([], terminal_reward=0.7, gamma=0.99)
    assert g == pytest.approx([0.7])


def test_returns_preserve_negative_regression():
    g = compute_per_turn_returns([0.5, -0.3], terminal_reward=0.0, gamma=1.0)
    # immediate' = [0.5, -0.3]; G1=-0.3; G0=0.5-0.3=0.2
    assert g == pytest.approx([0.2, -0.3])


def test_build_token_advantages_maps_turn_spans():
    per_turn = [[0.5, -0.2, 0.9]]  # rollout 0 has 3 actions
    # token turn-ids: query(0), turn1 tokens, tool(0), turn2 tokens, turn3 token
    turn_ids = [0, 0, 1, 1, 1, 0, 2, 2, 3]
    tok_adv = build_token_advantages(per_turn, turn_ids, rollout_id=0)
    assert tok_adv == pytest.approx([0.0, 0.0, 0.5, 0.5, 0.5, 0.0, -0.2, -0.2, 0.9])


def test_build_token_advantages_out_of_range_is_zero():
    per_turn = [[0.5]]
    tok_adv = build_token_advantages(per_turn, [0, 1, 2, 99], rollout_id=0)
    assert tok_adv == pytest.approx([0.0, 0.5, 0.0, 0.0])


def test_derive_turn_ids_from_mask():
    # query handled elsewhere; this is the response mask: two assistant spans separated by tool tokens
    mask = [1, 1, 0, 0, 1, 1, 1, 0]
    assert derive_turn_ids_from_mask(mask) == [1, 1, 0, 0, 2, 2, 2, 0]


def test_derive_turn_ids_all_assistant():
    assert derive_turn_ids_from_mask([1, 1, 1]) == [1, 1, 1]


def test_derive_turn_ids_leading_tool():
    assert derive_turn_ids_from_mask([0, 1, 0, 1]) == [0, 1, 0, 2]


def test_build_rollout_token_advantages_end_to_end():
    # 2 rollouts, 1 group. Rollout 0: 2 turns; rollout 1: 1 turn. terminal already in last reward slot.
    masks = [[1, 1, 0, 1], [1, 0]]  # r0: turn1(2 tok), tool, turn2(1 tok); r1: turn1, tool
    per_rollout_rewards = [[0.0, 1.0], [1.0]]  # r0: shaping 0 then terminal 1; r1: terminal 1
    tok_adv = build_rollout_token_advantages_return_pt(
        masks, per_rollout_rewards, num_samples_per_prompt=2, gamma=1.0, advantage_normalization_type="centered"
    )
    # returns: r0 G=[1,1] (G1=1, G0=0+1=1); r1 G=[1]. t0 group mean = (1+1)/2 = 1; t1 has only r0 active.
    # per-turn adv: r0 [0,0]; r1 [0]. All tokens -> 0.
    assert tok_adv[0] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert tok_adv[1] == pytest.approx([0.0, 0.0])


def test_build_rollout_token_advantages_discriminates():
    # Rollout 0 solves (terminal 1), rollout 1 fails (terminal 0). One turn each.
    masks = [[1, 1], [1, 1]]
    per_rollout_rewards = [[1.0], [0.0]]
    tok_adv = build_rollout_token_advantages_return_pt(
        masks, per_rollout_rewards, num_samples_per_prompt=2, gamma=1.0, advantage_normalization_type="centered"
    )
    # G0 = [1, 0], mean 0.5 -> adv per turn: r0 +0.5, r1 -0.5, on every assistant token.
    assert tok_adv[0] == pytest.approx([0.5, 0.5])
    assert tok_adv[1] == pytest.approx([-0.5, -0.5])


def test_pack_sequences_threads_per_token_advantages():
    """pack_sequences(per_token_advantages=...) must place each response token's advantage at its
    packed position (query tokens -> 0). Runs only where torch/vllm are importable (CI/cluster)."""
    pytest.importorskip("torch")
    try:
        import torch

        from open_instruct.rl_utils import pack_sequences
    except Exception as e:  # vllm/torch not importable here
        pytest.skip(f"rl_utils not importable: {type(e).__name__}")

    queries = [[10, 11], [12]]
    responses = [[100, 101, 102], [200, 201]]
    masks = [[1, 0, 1], [1, 1]]  # binary assistant/tool masks (unused for adv here)
    logprobs = [[-0.1, -0.2, -0.3], [-0.4, -0.5]]
    per_token_adv = [[0.5, 0.5, 0.9], [-0.3, -0.3]]
    ps = pack_sequences(
        queries=queries,
        responses=responses,
        masks=masks,
        pack_length=1000,
        pad_token_id=-1,
        vllm_logprobs=logprobs,
        mask_tool_use=False,
        per_token_advantages=per_token_adv,
    )
    assert ps.advantages is not None
    # Reconstruct expected per packed row: query positions 0, response positions = per_token_adv in order.
    for row_adv, row_prompt in zip(ps.advantages, ps.prompt_masks):
        adv = row_adv.tolist()
        is_prompt = row_prompt.tolist()
        # every prompt/query token has advantage 0
        for a, p in zip(adv, is_prompt):
            if p == 1:
                assert a == pytest.approx(0.0)
    # concatenation of response-position advantages (prompt==0) equals the flattened source, in order.
    flat_expected = [a for r in per_token_adv for a in r]
    flat_actual = []
    for row_adv, row_prompt in zip(ps.advantages, ps.prompt_masks):
        for a, p in zip(row_adv.tolist(), row_prompt.tolist()):
            if p == 0:
                flat_actual.append(a)
    assert flat_actual == pytest.approx(flat_expected)


@pytest.mark.skipif(not _HAVE_V72, reason="v7.2 grader codebase not present (set V72_CODEBASE)")
def test_returns_match_v72_oracle():
    """compute_per_turn_returns must equal v72 build_action_reward_assignments discounted returns."""
    if str(_V72_CODEBASE) not in sys.path:
        sys.path.insert(0, str(_V72_CODEBASE))
    try:  # v72 uses PEP 695 `type` aliases -> SyntaxError on Python < 3.12
        from training_data_collection.v72.contracts.progress import build_action_reward_assignments
    except (ImportError, SyntaxError) as e:
        pytest.skip(f"v72 contracts.progress not importable here: {type(e).__name__}")

    # Build synthetic progress events with known shaping rewards.
    import hashlib

    class _Ev:
        def __init__(self, shaping, idx):
            self.shaping_reward = shaping
            digest = hashlib.sha256(str(idx).encode()).hexdigest()
            self.progress_event_id = f"v72-progress-event:{digest}"

    gamma = 0.95
    shaping = [0.065, 0.13, 0.0, 0.24]
    terminal = 1.0
    events = tuple(_Ev(s, i) for i, s in enumerate(shaping))
    spans = tuple((i, i + 1) for i in range(len(shaping)))  # one non-empty token span per action

    try:
        assignments = build_action_reward_assignments(
            episode_id="v72-review-episode:" + hashlib.sha256(b"ep").hexdigest(),
            action_token_spans=spans,
            progress_events=events,
            terminal_reward=terminal,
            discount=gamma,
        )
    except TypeError:
        pytest.skip("v72 build_action_reward_assignments signature differs on this checkout")

    oracle_returns = [float(a.discounted_return) for a in assignments]
    ours = compute_per_turn_returns(shaping, terminal_reward=terminal, gamma=gamma)
    assert ours == pytest.approx(oracle_returns)


# ---------------------------------------------------------------------------
# return_pt (PPRS): discounted returns with a per-timestep group baseline.
# ---------------------------------------------------------------------------


def test_return_pt_zero_mean_per_turn():
    # The per-timestep baseline makes each turn's advantages zero-mean across the active group.
    returns = [[1.0, 0.4], [0.6, 0.2]]
    adv = compute_per_turn_advantages_pt(returns, num_samples_per_prompt=2, advantage_normalization_type="centered")
    assert adv[0] == pytest.approx([0.2, 0.1], abs=1e-9)
    assert adv[1] == pytest.approx([-0.2, -0.1], abs=1e-9)
    for t in range(2):
        assert adv[0][t] + adv[1][t] == pytest.approx(0.0, abs=1e-9)


def test_return_pt_removes_trajectory_baseline_offset():
    # Same returns, two baselines. A trajectory-level G_0 baseline leaves a systematic NEGATIVE mean on
    # the later turn; the per-timestep baseline zeroes every turn.
    returns = [[1.0, 0.4], [0.6, 0.2]]
    g0_mean = (returns[0][0] + returns[1][0]) / 2  # one group of 2
    traj = [[g - g0_mean for g in r] for r in returns]
    pt = compute_per_turn_advantages_pt(returns, num_samples_per_prompt=2, advantage_normalization_type="centered")
    traj_turn1_mean = (traj[0][1] + traj[1][1]) / 2
    pt_turn1_mean = (pt[0][1] + pt[1][1]) / 2
    assert traj_turn1_mean < -1e-6  # trajectory baseline: later-turn suppression present
    assert pt_turn1_mean == pytest.approx(0.0, abs=1e-9)  # per-timestep baseline: removed


def test_return_pt_standard_divides_by_per_timestep_std():
    returns = [[1.0, 0.4], [0.6, 0.2]]
    adv = compute_per_turn_advantages_pt(returns, num_samples_per_prompt=2, advantage_normalization_type="standard")
    # turn 0: vals [1.0,0.6] mu 0.8 std 0.2 -> ±0.2/0.2 = ±1 ; turn 1: vals [0.4,0.2] mu 0.3 std 0.1 -> ±1
    assert adv[0] == pytest.approx([1.0, 1.0], abs=1e-4)
    assert adv[1] == pytest.approx([-1.0, -1.0], abs=1e-4)


def test_return_pt_single_active_turn_centers_to_zero():
    # rollout 1 ends after turn 0; turns 1,2 have only rollout 0 active -> centered advantage 0.
    adv = compute_per_turn_advantages_pt([[1.0, 0.7, 0.3], [0.8]], num_samples_per_prompt=2)
    assert adv[0][1] == pytest.approx(0.0)
    assert adv[0][2] == pytest.approx(0.0)


def test_return_pt_divisibility_error():
    with pytest.raises(ValueError):
        compute_per_turn_advantages_pt([[1.0], [0.5], [0.2]], num_samples_per_prompt=2)


def test_return_pt_rejects_bad_norm_type():
    with pytest.raises(ValueError):
        compute_per_turn_advantages_pt([[1.0], [0.5]], num_samples_per_prompt=2, advantage_normalization_type="maxrl")


def test_return_pt_end_to_end_scatter():
    # discounted returns (gamma=1) then per-timestep baseline then scatter to token spans.
    mask0 = [1, 1, 0, 1, 0, 1]  # turns 1,1,-,2,-,3
    mask1 = [1, 0, 1]  # turns 1,-,2
    r0 = [0.1, 0.2, 0.7]  # returns backward -> [1.0, 0.9, 0.7]
    r1 = [0.3, 0.5]  # returns backward -> [0.8, 0.5]
    tok = build_rollout_token_advantages_return_pt([mask0, mask1], [r0, r1], num_samples_per_prompt=2, gamma=1.0)
    # per-turn (centered): t0 [.1,-.1], t1 [.2,-.2], t2 rollout0-only -> 0
    assert tok[0] == pytest.approx([0.1, 0.1, 0.0, 0.2, 0.0, 0.0], abs=1e-9)
    assert tok[1] == pytest.approx([-0.1, 0.0, -0.2], abs=1e-9)


def test_return_pt_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_rollout_token_advantages_return_pt([[1, 1]], [[0.5], [0.5]], num_samples_per_prompt=2, gamma=1.0)
