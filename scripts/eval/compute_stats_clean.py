#!/usr/bin/env python3
"""Exception-aware pass@k over a Harbor jobs dir.

Unlike scripts/eval/compute_stats.py (which counts EVERY graded trial, so an infra-failed
trial with reward 0 looks like a wrong answer), this drops clear infra failures
(API/serving errors, grader hangs) from numerator AND denominator, reports
AgentTimeoutError trials separately, and drops them too under --drop-timeouts
(the fair cross-hardware view). eval_tb21.sh and eval_tb4_cm.sh run it to write
metrics_clean.json next to the unchanged metrics.json.

Usage:
    python scripts/eval/compute_stats_clean.py <jobs_dir> [<jobs_dir> ...] [--json-output metrics_clean.json] [-k 5]

Schema-tolerant: Harbor writes one result.json per trial. We recursively find
every result.json, extract (task_id, reward) with a few fallback key paths, then
aggregate by task. pass@k uses the standard unbiased estimator
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = trials for the task and c = number that passed (reward >= threshold).
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PASS_THRESHOLD = 1.0  # a trial "passes" when reward >= this

# Exception types that mean the trial failed for INFRA reasons, not model quality.
# These trials carry a reward (usually 0.0) that compute_stats.py counts as a
# genuine model failure -- which biases scores DOWN, and biases them MORE on slower
# hardware (in our runs, A100 hit the agent-timeout ceiling ~3.5x as often as H100).
# We drop them from BOTH numerator and denominator, exactly like null-verifier trials.
#   * "clear infra" (always dropped): serving/API errors + the grader hanging.
#   * AgentTimeoutError is AMBIGUOUS (a too-slow GPU vs a model that genuinely ran out of
#     budget), so it is dropped only under --drop-timeouts, and always reported separately.
INFRA_EXC_CLEAR = ("UnknownApiError", "VerifierTimeoutError", "APIError", "APIConnectionError",
                   "APITimeoutError", "ServiceUnavailable", "InternalServerError")
INFRA_EXC_TIMEOUT = ("AgentTimeoutError",)


def _exc_type(d):
    exc = d.get("exception_info")
    return exc.get("exception_type") if isinstance(exc, dict) else None


def _dig(d, *paths):
    """Return the first present value among dotted key paths, else None."""
    for path in paths:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def load_trial(result_path: Path):
    """Return (task_id, reward_float, kind) or None if unparseable.

    kind in {"clean", "infra", "timeout"}:
      * clean   -- graded trial with no infra exception; a real model signal.
      * infra   -- clear infra failure (serving/API error, grader hang); reward is junk.
      * timeout -- AgentTimeoutError; ambiguous (slow HW vs out-of-budget model).
    A trial with no reward at all (null-verifier) still returns None -> dropped, as before.
    """
    try:
        d = json.loads(result_path.read_text())
    except Exception:
        return None
    # harbor 0.20.0: per-trial result.json has task_name (str) + task_id (dict).
    task_id = _dig(d, "task_name", "task_id.name", "task.name", "instance_id")
    if task_id is None:
        # fall back to the directory name two levels up (jobs/<name>/<task>/<trial>/result.json)
        task_id = result_path.parent.name
        if task_id.isdigit():
            task_id = result_path.parent.parent.name
    reward = _dig(
        d,
        "verifier_result.rewards.reward",  # observed Harbor schema
        "verifier_result.reward",
        "reward",
        "rewards.reward",
        "score",
    )
    if reward is None:
        passed = _dig(d, "is_resolved", "resolved", "passed", "success")
        reward = 1.0 if passed else 0.0 if passed is not None else None
    if reward is None:
        return None  # null-verifier: no gradable result -> drop (numerator + denominator)
    exc = _exc_type(d) or ""
    kind = "clean"
    if any(e in exc for e in INFRA_EXC_CLEAR):
        kind = "infra"
    elif any(e in exc for e in INFRA_EXC_TIMEOUT):
        kind = "timeout"
    try:
        return str(task_id), float(reward), kind
    except (TypeError, ValueError):
        return None


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for a task with n trials, c passing."""
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_dir", type=Path, nargs="+",
                    help="jobs dir(s); extra dirs are POOLED (e.g. two runs of the same checkpoint with an "
                         "identical model/harness/dataset/k); trials de-duplicated by trial dir name")
    ap.add_argument("--json-output", type=Path, default=None)
    ap.add_argument("-k", type=int, default=None, help="k for pass@k (default: max trials seen)")
    ap.add_argument("--count-infra", action="store_true",
                    help="count clear infra-failure trials (API/serving errors, grader hangs) as "
                         "reward-0 instead of dropping them (as compute_stats.py does)")
    ap.add_argument("--drop-timeouts", action="store_true",
                    help="also drop AgentTimeoutError trials (recommended when comparing across "
                         "hardware: A100 hits the ceiling far more often than H100)")
    args = ap.parse_args()

    for d in args.jobs_dir:
        if not d.exists():
            print(f"ERROR: {d} does not exist", file=sys.stderr)
            return 2

    per_task_rewards: dict[str, list[float]] = defaultdict(list)
    dropped = {"infra": 0, "timeout": 0}
    files, seen_trial_dirs = [], set()
    for d in args.jobs_dir:
        for f in sorted(d.rglob("result.json")):
            if f.parent.name in seen_trial_dirs:      # same trial present in more than one jobs dir
                continue
            seen_trial_dirs.add(f.parent.name)
            files.append(f)
    for f in files:
        parsed = load_trial(f)
        if not parsed:
            continue
        task, reward, kind = parsed
        if kind == "infra" and not args.count_infra:
            dropped["infra"] += 1
            continue
        if kind == "timeout" and args.drop_timeouts:
            dropped["timeout"] += 1
            continue
        per_task_rewards[task].append(reward)

    if dropped["infra"] or dropped["timeout"]:
        parts = []
        if dropped["infra"]:
            parts.append(f"{dropped['infra']} clear-infra")
        if dropped["timeout"]:
            parts.append(f"{dropped['timeout']} agent-timeouts")
        print(f"dropped infra-failure trials: {', '.join(parts)} "
              f"(--count-infra keeps clear-infra; --drop-timeouts also drops timeouts)",
              file=sys.stderr)

    if not per_task_rewards:
        print(f"ERROR: no parseable result.json under {args.jobs_dir}", file=sys.stderr)
        print("       (check the Harbor output schema and adjust _dig paths)", file=sys.stderr)
        return 3

    all_rewards = [r for rs in per_task_rewards.values() for r in rs]
    n_total = len(all_rewards)
    mean = sum(all_rewards) / n_total
    var = sum((r - mean) ** 2 for r in all_rewards) / n_total if n_total > 1 else 0.0
    std = math.sqrt(var)
    sem = std / math.sqrt(n_total) if n_total else 0.0

    k = args.k or max(len(rs) for rs in per_task_rewards.values())
    per_task = {}
    pass_k_vals = []
    for task, rs in sorted(per_task_rewards.items()):
        n = len(rs)
        c = sum(1 for r in rs if r >= PASS_THRESHOLD)
        pk = pass_at_k(n, c, k)
        pass_k_vals.append(pk)
        per_task[task] = {
            "trials": n,
            "passed": c,
            "mean_reward": sum(rs) / n,
            f"pass@{k}": pk,
        }
    macro_pass_k = sum(pass_k_vals) / len(pass_k_vals)

    summary = {
        "jobs_dir": str(args.jobs_dir),
        "n_tasks": len(per_task_rewards),
        "n_trials_total": n_total,
        "k": k,
        "mean_reward": mean,
        "std_reward": std,
        "sem_reward": sem,
        f"pass@{k}_macro": macro_pass_k,
        "per_task": per_task,
    }

    print(f"tasks={summary['n_tasks']}  trials={n_total}  k={k}")
    print(f"mean_reward = {mean:.4f} ± {sem:.4f} (sem)")
    print(f"pass@{k} (macro over tasks) = {macro_pass_k:.4f}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
