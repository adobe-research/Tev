#!/usr/bin/env python3
"""Check that a grader codebase matches a dataset: every task package must load and validate against it.

For each task in DATASET_DIR/task-data (or task-data.tar.gz, extracted to a temp dir) this builds the exact
host-side grader the training env uses (training/open-instruct/open_instruct/environments/v72_grader.py) from
the task's authority/package and the grader codebase, then grades an EMPTY workspace. A package whose family
implementation does not match the codebase fails at load ("authority is stale" / validation error) — during
training such a task would silently fall back to terminal-only reward, so fix the mismatch before training.

Expected on the shipped example: 1/1 packages load; empty workspace -> progress 0.0 / terminal 0.0.
Note: loading a package normalizes it in place (as in training): public inputs are made read-only and missing
empty directories are recreated.

Usage (with the training venv, Python >= 3.12):
    python scripts/check_grader.py --dataset-dir examples/data/warehouse_etl_example --codebase examples/grader
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tarfile
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GRADER_PY = REPO / "training/open-instruct/open_instruct/environments/v72_grader.py"


def _grader_cls():
    spec = importlib.util.spec_from_file_location("v72_grader", GRADER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.V72Grader


def _check(args: tuple[str, str]) -> dict:
    task_dir, codebase = Path(args[0]), args[1]
    cfg = json.loads((task_dir / "dense_config.json").read_text())
    rec = {"task": task_dir.name, "family": cfg["family_revision_id"], "grade_mode": cfg["grade_mode"]}
    try:
        grader = _grader_cls()(task_dir / "authority" / "package", codebase=codebase)
    except Exception as e:  # package does not validate against this codebase
        rec["load"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec
    rec["load"] = "ok"
    empty = Path(tempfile.mkdtemp(prefix="check-grader-empty-"))
    for name, fn in (("progress", lambda d: grader.progress(d)["potential"]), ("terminal", grader.terminal)):
        try:
            rec[name] = round(float(fn(empty)), 4)
        except Exception as e:
            rec[name] = f"error: {type(e).__name__}"
    grader.close()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-dir", required=True, help="dir with task-data/ or task-data.tar.gz")
    ap.add_argument("--codebase", required=True, help="grader codebase (contains training_data_collection/v72)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    td = Path(args.dataset_dir) / "task-data"
    if not any(td.glob("*/dense_config.json")):
        tarball = Path(args.dataset_dir) / "task-data.tar.gz"
        td = Path(tempfile.mkdtemp(prefix="check-grader-td-"))
        with tarfile.open(tarball) as tar:
            tar.extractall(td, filter="tar")
    tasks = sorted(p.parent for p in td.glob("*/dense_config.json"))
    if not tasks:
        raise SystemExit(f"no tasks found under {args.dataset_dir}")
    codebase = str(Path(args.codebase).resolve())
    with ProcessPoolExecutor(args.jobs) as ex:
        recs = list(ex.map(_check, [(str(t), codebase) for t in tasks]))

    by_family: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        if r["load"] != "ok":
            verdict = "LOAD-FAIL"
        elif r["grade_mode"] == "terminal-only":
            verdict = "terminal-only"
        else:
            verdict = "ok" if r.get("progress") == 0.0 and r.get("terminal") == 0.0 else "UNEXPECTED"
        by_family[r["family"]][verdict] += 1
    for family, counts in sorted(by_family.items()):
        print(f"{family:<50} {dict(counts)}")
    failed = [r for r in recs if r["load"] != "ok"]
    print(f"\n{len(recs) - len(failed)}/{len(recs)} task packages load against {codebase}")
    for r in failed[:10]:
        print(f"  LOAD-FAIL {r['task']}: {r['load']}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
