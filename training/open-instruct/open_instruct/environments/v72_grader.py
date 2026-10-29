"""Host-side dense-reward grading driver for Tev-Gym tasks (grader contracts v7.2, package `v72`).

Tested to give Φ=1.0 / terminal=1.0 on each task family's reference output and 0/0 on an empty
workspace (see scripts/check_grader.py for the package-vs-codebase consistency check).

Given a built task PACKAGE dir + a candidate OUTPUT dir it produces the dense progress potential
Φ (+ per-node vectors) and the reward-contract terminal score, using the real
`training_data_collection.v72` APIs (the grader codebase; examples/grader in this repo).

Runs HOST-SIDE in the env actor: the authority (grading package) stays on the host and is never uploaded
into the agent container. `training_data_collection.v72` must be importable (Python >= 3.12, pydantic 2.x);
v72 is imported LAZILY in `V72Grader.__init__` so importing this module never fails — the env catches
instantiation failures and degrades to terminal-only.
"""

from __future__ import annotations

import inspect
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

RUNTIME_IDENTITY = "v72-dense-reward-bridge.v1"

_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _ensure_codebase_on_path(codebase: str | None) -> None:
    root = codebase or os.environ.get("V72_CODEBASE_PATH")  # the env passes v72_codebase_path explicitly
    if root and root not in sys.path and Path(root).is_dir():
        sys.path.insert(0, root)


def normalize_package(package_dir: str | os.PathLike) -> None:
    """Idempotently repair a task package whose empty dirs and read-only public-input bits were lost in
    transit (git and object stores keep no empty directories, and archives/copies may drop POSIX modes).
    ``validate_task_package`` requires ``public/output/`` + every required directory role to exist, and
    ``public/{prompt.md,input,software}`` (dirs included) to carry no write bits."""
    pkg = Path(package_dir)
    public = pkg / "public"
    if not public.is_dir():
        return
    (public / "output").mkdir(exist_ok=True)
    spec_path = pkg / "private" / "authority" / "public_specification.json"
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            for role in spec.get("public_inputs", []):
                if role.get("path_kind") == "directory" and role.get("root") and role.get("path"):
                    (public / role["root"] / role["path"]).mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError):
            pass
    targets: list[Path] = []
    for name in ("prompt.md", "input", "software"):
        p = public / name
        if not p.exists():
            continue
        targets.append(p)
        if p.is_dir():
            targets.extend(p.rglob("*"))
    for path in targets:
        try:
            mode = path.stat().st_mode
            if mode & _WRITE_BITS:
                path.chmod(mode & ~_WRITE_BITS)
        except OSError:
            pass


def _resolve_verifier(package_dir: Path, family_package: str, importlib):  # noqa: ANN001
    """Callable for the package's declared terminal verifier (manifest `family_revision.verifier_authority`;
    falls back to the family's `run_verifier_process`)."""
    entrypoint = None
    manifest_path = package_dir / "package_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entrypoint = (manifest.get("family_revision", {}).get("verifier_authority", {}) or {}).get("entrypoint")
        except (OSError, ValueError):
            entrypoint = None
    if entrypoint:
        module_name, _, func_name = entrypoint.partition(":")
        return getattr(importlib.import_module(module_name), func_name)
    return getattr(importlib.import_module(f"{family_package}.verifier"), "run_verifier_process")


class V72Grader:
    """Loaded package + persistent ProgressEvaluator; grade a candidate output repeatedly."""

    def __init__(
        self,
        package_dir: str | os.PathLike,
        codebase: str | None = None,
        snapshot_root: str | os.PathLike | None = None,
    ) -> None:
        _ensure_codebase_on_path(codebase)
        import importlib

        from training_data_collection.common.identity import content_id
        from training_data_collection.v72.families.registry import FamilyRegistry
        from training_data_collection.v72.generation.service import validate_task_package
        from training_data_collection.v72.runtime.progress import ProgressEvaluator, SnapshotStore

        self._content_id = content_id
        self._package_dir = Path(package_dir).resolve(strict=True)
        normalize_package(self._package_dir)  # self-heal output/ + read-only bits lost in transit
        pkg = validate_task_package(self._package_dir)
        if pkg.progress_contract is None or pkg.reward_contract is None:
            raise ValueError(f"package {self._package_dir} has no dense-reward authority")

        self.family_revision_id: str = pkg.recipe.family_revision_id
        self._task_id: str = pkg.task_manifest.task_id
        self._public_input_root = pkg.root / "public" / "input"
        self._private_root = pkg.root / "private"
        self._reward_contract = pkg.reward_contract

        registration = FamilyRegistry().registration(self.family_revision_id)
        family_package = registration.family_package
        progress_mod = importlib.import_module(f"{family_package}.progress")
        # Terminal entrypoint differs by version (v2 run_verifier_process(payload)->dict vs v1
        # verify_<family>(task_id, reward_contract, public_input_root, sealed_output_root)->RewardResult).
        self._verifier_fn = _resolve_verifier(self._package_dir, family_package, importlib)

        contract = pkg.progress_contract
        self.shaping_scale: float = float(contract.shaping_scale)
        checker_registry = progress_mod.build_progress_checker_registry(contract)

        self._snapshot_tmp = None
        if snapshot_root is None:
            self._snapshot_tmp = tempfile.TemporaryDirectory(prefix="v72-dense-snap-")
            snapshot_root = Path(self._snapshot_tmp.name) / "snapshots"
        self._evaluator = ProgressEvaluator(
            contract=contract,
            checker_registry=checker_registry,
            snapshot_store=SnapshotStore(Path(snapshot_root)),
            public_root=self._public_input_root,
            private_root=self._private_root,
            target_authority_id=pkg.world.world_id,
            runtime_identity=RUNTIME_IDENTITY,
        )
        self._eval_counter = 0

    def progress(self, output_dir: str | os.PathLike) -> dict:
        """{potential, effective, raw}. potential Φ ∈ [0,1]. Raises if output_dir is missing."""
        output_dir = Path(output_dir).resolve(strict=True)
        self._eval_counter += 1
        episode_id = self._content_id(
            "v72-bridge-episode", {"task_id": self._task_id, "n": self._eval_counter}
        )
        snapshot = self._evaluator.capture_initial(episode_id=episode_id, output_root=output_dir)
        checkpoint = self._evaluator.evaluate(snapshot)
        potential = checkpoint.potential
        return {
            "potential": 0.0 if potential is None else float(potential),
            "effective": {k: (None if v is None else float(v)) for k, v in dict(checkpoint.effective_vector).items()},
            "raw": {k: (None if v is None else float(v)) for k, v in dict(checkpoint.raw_vector).items()},
        }

    def _invoke_verifier(self, output_dir: Path):
        """Run the declared verifier, dispatching on its signature (v2 single-param
        `run_verifier_process(payload)->dict` vs v1 direct
        `verify_<family>(task_id, reward_contract, public_input_root, sealed_output_root)->RewardResult`)."""
        params = list(inspect.signature(self._verifier_fn).parameters)
        if len(params) == 1:
            return self._verifier_fn(
                {
                    "task_id": self._task_id,
                    "reward_contract": self._reward_contract.model_dump(mode="json"),
                    "public_input_root": str(self._public_input_root),
                    "sealed_output_root": str(output_dir),
                }
            )
        available = {
            "task_id": self._task_id,
            "reward_contract": self._reward_contract,
            "public_input_root": self._public_input_root,
            "input_root": self._public_input_root,
            "sealed_output_root": output_dir,
            "output_root": output_dir,
        }
        return self._verifier_fn(**{name: available[name] for name in params if name in available})

    def terminal_full(self, output_dir: str | os.PathLike) -> dict:
        """One verifier run → {"total", "component_values", "component_pass_mean"}.

        * ``total``: reward-contract score ∈ [0,1] (Σ weightᵢ·componentᵢ, zeroed by the hard gate).
        * ``component_values``: the per-component continuous scores (dict).
        * ``component_pass_mean``: equal-weight BOOLEAN mean — the fraction of components that are
          fully correct (score ≥ 1−eps), UNGATED; logged only, never used as reward. ``None`` if the
          verifier exposes no ``component_values``.
        """
        output_dir = Path(output_dir).resolve(strict=True)
        result = self._invoke_verifier(output_dir)
        get = (result.get if isinstance(result, dict) else lambda k: getattr(result, k, None))
        total = get("total")
        total = max(0.0, min(1.0, float(total))) if total is not None else 0.0
        component_values = get("component_values")
        component_pass_mean = None
        if component_values:
            try:
                vals = list(dict(component_values).values())
                if vals:
                    component_pass_mean = sum(1.0 if float(v) >= 1.0 - 1e-9 else 0.0 for v in vals) / len(vals)
            except (TypeError, ValueError):
                component_pass_mean = None
        return {"total": total, "component_values": component_values, "component_pass_mean": component_pass_mean}

    def terminal(self, output_dir: str | os.PathLike) -> float:
        """Reward-contract terminal score in [0,1] (weighted, hard-gated)."""
        return self.terminal_full(output_dir)["total"]

    def close(self) -> None:
        if self._snapshot_tmp is not None:
            self._snapshot_tmp.cleanup()
            self._snapshot_tmp = None
