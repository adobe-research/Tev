#!/usr/bin/env python3
"""Summarize a K-rollout ALE computing_math eval (the layout eval_ale_cm.sh writes).

<out_root>/r1 .. <out_root>/rK are independent ale_run output roots, one per rollout. Each holds
<agent>/<model>/computing_math__<task>/v0/<ts>/run.json; the latest run.json per task wins (a task that
was re-run on resume counts once). ALE scores are in [0, 1].

Reported over the 18 tasks:
  pass@1      mean over rollouts of the per-rollout mean score, +/- sem (population std over rollouts
              / sqrt(K), as in compute_stats.py)
  pass@K      best-of-K: mean over tasks of the max score across the K rollouts
A task with no run.json in a rollout, or with a null score, counts as 0 and is listed so an incomplete
run is visible.

Usage:
    python scripts/eval/ale_cm/summarize_ale_cm.py results/ale-cm-Qwen3.8-27B/rollouts
    python scripts/eval/ale_cm/summarize_ale_cm.py <out_root> -k 5 --json-output metrics.json
"""

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TASKS = HERE / "computing_math.txt"
_SLUG_RE = re.compile(r"computing_math__([^/]+)/v\d+/")


def load_tasks(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.rsplit("/", 1)[-1])
    return out


def task_of(run_json: Path, d: dict) -> str | None:
    p = ((d.get("task") or {}).get("path") or "").rstrip("/")
    if p:
        return p.rsplit("/", 1)[-1]
    m = _SLUG_RE.search(str(run_json))
    return m.group(1) if m else None


def load_rollout(rdir: Path) -> dict[str, dict]:
    """{task: {"score": float|None, "status": str}} from the latest run.json per task."""
    latest: dict[str, tuple[str, dict]] = {}
    for rj in rdir.rglob("run.json"):
        try:
            d = json.loads(rj.read_text())
        except (OSError, ValueError):
            continue
        t = task_of(rj, d)
        if t is None:
            continue
        ts = d.get("timestamp_utc") or ""
        if t not in latest or ts > latest[t][0]:
            latest[t] = (ts, d)
    out = {}
    for t, (_, d) in latest.items():
        s = d.get("score")
        out[t] = {
            "score": float(s) if isinstance(s, (int, float)) else None,
            "status": d.get("status") or "?",
        }
    return out


def score(rollout: dict[str, dict], task: str) -> float:
    s = (rollout.get(task) or {}).get("score")
    return s if s is not None else 0.0


def report(rollouts: list[dict], tasks: list[str]) -> dict:
    k = len(rollouts)
    runs = [sum(score(r, t) for t in tasks) / len(tasks) for r in rollouts]
    mu = sum(runs) / k
    sd = (sum((x - mu) ** 2 for x in runs) / k) ** 0.5
    best = sum(max(score(r, t) for r in rollouts) for t in tasks) / len(tasks)
    return {
        "n_tasks": len(tasks),
        "k": k,
        "pass_at_1": mu,
        "std": sd,
        "sem": sd / math.sqrt(k),
        f"pass_at_{k}": best,
        "run_scores": runs,
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize a K-rollout ALE computing_math eval.")
    ap.add_argument("out_root", type=Path, help="Directory holding the r1..rK rollout roots")
    ap.add_argument("-k", type=int, default=None, help="Number of rollouts (default: count r* dirs)")
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="Task list (.txt)")
    ap.add_argument("--json-output", type=Path, default=None)
    args = ap.parse_args()

    k = args.k or len([p for p in args.out_root.glob("r[0-9]*") if p.is_dir()])
    if k < 1:
        raise SystemExit(f"no rollout dirs (r1..rK) under {args.out_root}")
    tasks = load_tasks(args.tasks)
    rollouts = [load_rollout(args.out_root / f"r{i}") for i in range(1, k + 1)]

    w = max(len(t) for t in tasks)
    print(f"{'task':{w}s}  " + "  ".join(f"{'r' + str(i + 1):>5s}" for i in range(k)) + "   mean    max")
    missing, unscored = [], []
    for t in tasks:
        cells = []
        for i, r in enumerate(rollouts):
            e = r.get(t)
            if e is None:
                cells.append(f"{'-':>5s}")
                missing.append(f"r{i + 1}:{t}")
            elif e["score"] is None:
                cells.append(f"{'x':>5s}")
                unscored.append(f"r{i + 1}:{t}({e['status']})")
            else:
                cells.append(f"{e['score']:5.2f}")
        xs = [score(r, t) for r in rollouts]
        print(f"{t:{w}s}  " + "  ".join(cells) + f"  {sum(xs) / k:5.3f}  {max(xs):5.2f}")
    print("('-' = no run.json yet, 'x' = run finished without a score; both count as 0)")
    print()

    m = report(rollouts, tasks)
    metrics = {"out_root": str(args.out_root), **m}
    rr = ",".join(f"{x:.3f}" for x in m["run_scores"])
    print(
        f"ALE-CM ({len(tasks)} tasks)  pass@1 = {m['pass_at_1']:.3f} +/- {m['sem']:.3f} sem  "
        f"(+/-{m['std']:.3f} std; per-rollout: {rr})  pass@{k}(best-of) = {m[f'pass_at_{k}']:.3f}"
    )
    metrics["missing"] = missing
    metrics["unscored"] = unscored
    if missing:
        print(f"\nINCOMPLETE: {len(missing)} (rollout, task) pairs have no run.json yet; re-run to fill them.")
    if unscored:
        print(f"{len(unscored)} runs finished without a score: {', '.join(unscored)}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
