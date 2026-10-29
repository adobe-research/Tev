"""SWERL Vanillux DENSE sandbox — per-sample Docker tasks with Tev-Gym dense (PPRS) reward.

Extends :class:`SWERLVanilluxSandboxEnv` (single ``bash`` tool, ``/app`` workspace) with per-turn
progress-potential reward shaping (PPRS) from the task's latent-world progress contract, plus a
host-side terminal reward.

How it works (all grading HOST-SIDE: the authority package never enters the agent container):
  * ``reset`` builds a :class:`V72Grader` from ``{task_data_dir}/{task_id}/authority/package`` (shipped
    in the dataset's task-data), and records the baseline potential Φ₀ = 0
    (empty output).
  * after each state-changing ``bash`` turn (``grade_mode == "live"``) the env extracts ``/app/output``
    from the container to a host temp dir, grades it → potential Φₜ, and emits a potential-based
    shaping reward ``StepResult.reward = λ·(γ·Φₜ − Φₜ₋₁)`` (λ = contract ``shaping_scale``, 0.25 in
    Tev-Gym, overridable with ``shaping_beta``; γ = ``shaping_gamma``, default 1.0 → the sum telescopes
    to λ·Φ_final). A content hash of
    ``/app/output`` gates re-grading so read-only turns are cheap.
  * on submit (``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``) the terminal reward is the reward-contract
    score, graded host-side. No ``tests/`` upload, no in-container v72.

Training pairs it with ``--reward_aggregator last --per_turn_credit --per_turn_credit_gamma 1.0`` (PPRS):
the per-turn PBRS rewards reach the learner through ``rollout_states[i]["rewards"]`` and are turned into
per-token advantages by ``open_instruct.dense_credit``. The env is a drop-in for ``swerl_vanillux_sandbox``
— a task not shipping ``authority/package`` (or with ``grade_mode == "terminal-only"``) still runs,
contributing only the terminal reward.

``grade_mode`` (per task, from the task's ``dense_config.json``; the ``grade_mode`` arg is the default):
``live`` (per-turn shaping + host-side terminal), ``terminal-vector`` (grade once at submit, no shaping),
``terminal-only`` (no progress grading at all — the safe fallback).
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from open_instruct import logger_utils

from .base import BaseEnvConfig, RLEnvironment, StepResult
from .swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv

logger = logger_utils.setup_logger(__name__)


class SWERLVanilluxDenseSandboxEnv(SWERLVanilluxSandboxEnv):
    """Vanillux sandbox with Tev-Gym dense (progress-potential) reward shaping."""

    config_name = "swerl_vanillux_dense"

    def __init__(
        self,
        *args: Any,
        grade_mode: str = "live",
        require_grader: bool = False,
        shaping_gamma: float = 1.0,
        shaping_beta: float | None = None,
        v72_codebase_path: str | None = None,
        authority_subdir: str = "authority/package",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._grade_mode = grade_mode
        self._require_grader = require_grader
        self._active_grade_mode = grade_mode
        self._shaping_gamma = float(shaping_gamma)
        self._shaping_beta_override = shaping_beta
        self._v72_codebase_path = v72_codebase_path
        self._authority_subdir = authority_subdir
        self._grader = None
        self._beta = 0.0
        self._prev_potential = 0.0
        self._prev_output_hash: str | None = None
        self._final_progress: dict | None = None
        self._terminal_raw = 0.0
        self._terminal_bool_mean = 0.0
        self._progress_grade_calls = 0
        self._grading_seconds = 0.0

    # ------------------------------------------------------------------
    # Reset: build the host-side grader
    # ------------------------------------------------------------------
    def _do_reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        result, tools = super()._do_reset(task_id, **kwargs)
        self._teardown_grader()
        self._beta = 0.0
        self._prev_potential = 0.0
        self._prev_output_hash = None
        self._final_progress = None
        self._terminal_raw = 0.0
        self._terminal_bool_mean = 0.0
        self._progress_grade_calls = 0
        self._grading_seconds = 0.0

        # Per-task grade_mode override (mixed corpora): the dense_config.json each task ships.
        grade_mode = self._grade_mode
        cfg_path = (
            os.path.join(self._task_data_dir, self._task_id, "dense_config.json")
            if self._task_data_dir and self._task_id
            else None
        )
        if cfg_path and os.path.isfile(cfg_path):
            try:
                import json

                with open(cfg_path, encoding="utf-8") as f:
                    grade_mode = json.load(f).get("grade_mode", grade_mode)
            except Exception:
                pass
        self._active_grade_mode = grade_mode

        # Build the grader for EVERY mode when authority is present: Tev-Gym tasks have no in-container
        # test.sh, so the terminal reward is graded host-side too. grade_mode only controls whether
        # per-turn shaping is emitted (live) vs not (terminal-vector/terminal-only).
        authority = self._authority_dir()
        if authority is None:
            # Missing authority dir is treated as recoverable (degrade this rollout), NOT fatal: a
            # single task's missing/unextracted authority must never kill the whole training job.
            logger.warning(
                "[%s] no authority package found (grade_mode=%s); terminal reward 0 for this rollout.",
                self._task_id, grade_mode,
            )
            return result, tools
        try:
            from .v72_grader import V72Grader

            self._grader = V72Grader(authority, codebase=self._v72_codebase_path)
            self._beta = (
                self._shaping_beta_override
                if self._shaping_beta_override is not None
                else self._grader.shaping_scale
            )
        except Exception as e:
            self._grader = None
            # `require_grader` is fail-LOUD only for a genuine misconfig (v7.2 not importable at all),
            # caught at launch. Transient/environmental failures — a node restart that wiped the local
            # venv, an FS race, a package-validation "runtime dependency discovery" miss — MUST NOT kill
            # the job (otherwise a single node preemption becomes a failed run).
            # Degrade this rollout to terminal-only and continue; the loss is visible via the
            # progress_potential_final metric (0 => grading unavailable).
            if self._require_grader and _grader_error_is_fatal(e):
                raise RuntimeError(
                    f"[{self._task_id}] V72Grader init failed (v7.2 not importable) and require_grader=True: "
                    f"{type(e).__name__}: {e}"
                ) from e
            logger.warning(
                "[%s] V72Grader init failed (%s: %s); falling back to terminal-only for this rollout "
                "(transient/environmental — not fatal).",
                self._task_id,
                type(e).__name__,
                e,
            )
        return result, tools

    def _authority_dir(self) -> str | None:
        if not self._task_data_dir or not self._task_id:
            return None
        path = os.path.join(self._task_data_dir, self._task_id, self._authority_subdir)
        return path if os.path.isdir(path) else None

    # ------------------------------------------------------------------
    # Per-turn shaping
    # ------------------------------------------------------------------
    def _execute_bash(self, args: dict) -> StepResult:
        result = super()._execute_bash(args)
        # Submit path is handled by _run_tests (overridden below) and returns done=True.
        if result.done or self._grader is None or self._active_grade_mode != "live":
            return result
        reward = self._compute_shaping()
        if reward is not None:
            result.reward = float(result.reward or 0.0) + reward
            meta = dict(result.metadata or {})
            meta["progress_potential"] = self._prev_potential
            meta["shaping_reward"] = reward  # per-turn PBRS delta
            result.metadata = meta
        return result

    def _compute_shaping(self) -> float | None:
        """Per-turn reward for the current /app/output: potential-based shaping λ·(γ·Φₜ − Φₜ₋₁); None if
        ungradeable."""
        t0 = time.perf_counter()
        output_dir = self._extract_output()
        try:
            if output_dir is None:
                phi = 0.0
            else:
                h = _tree_hash(output_dir)
                if h == self._prev_output_hash:
                    phi = self._prev_potential  # unchanged -> reuse (no expensive re-grade)
                else:
                    self._prev_output_hash = h
                    self._progress_grade_calls += 1
                    try:
                        phi = float(self._grader.progress(output_dir)["potential"])
                    except Exception as e:
                        logger.warning("[%s] progress grade failed: %s: %s", self._task_id, type(e).__name__, e)
                        return None
            reward = self._beta * (self._shaping_gamma * phi - self._prev_potential)  # PBRS delta
            self._prev_potential = phi
            return reward
        finally:
            _rmtree(output_dir)
            self._grading_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Terminal reward (host-side; overrides the in-container test.sh path)
    # ------------------------------------------------------------------
    def _run_tests(self) -> StepResult:
        if self._grader is None:
            # No dense grader: fall back to the inherited in-container test.sh path if tests exist,
            # else zero terminal reward.
            if self._tests_dir is not None:
                return super()._run_tests()
            return StepResult(result="No grader available; terminal reward 0.0", reward=0.0, done=True)

        t0 = time.perf_counter()
        output_dir = self._extract_output()
        try:
            terminal = 0.0
            component_pass_mean = None
            progress_note = ""
            if output_dir is not None:
                try:
                    info = self._grader.terminal_full(output_dir)
                    terminal = info["total"]
                    component_pass_mean = info["component_pass_mean"]
                except Exception as e:  # host-side terminal ungradeable (e.g. a verifier that must run in the sandbox)
                    logger.warning("[%s] terminal grade failed: %s: %s", self._task_id, type(e).__name__, e)
                    terminal = 0.0
                self._progress_grade_calls += 1
                try:
                    self._final_progress = self._grader.progress(output_dir)
                    progress_note = f" progress_potential={self._final_progress['potential']:.4f}"
                except Exception:
                    self._final_progress = None
            self._terminal_raw = float(terminal)
            # Boolean mean (fraction of components fully correct, ungated) — logged only; falls back to the
            # binarized total if the verifier exposes no component vector.
            self._terminal_bool_mean = float(
                component_pass_mean
                if component_pass_mean is not None
                else (1.0 if self._terminal_raw >= 1.0 - 1e-9 else 0.0)
            )
            reward = self._terminal_raw  # the continuous weighted .total of the reward contract
            observation = f"Test execution complete. Terminal reward: {reward}{progress_note}"
            return StepResult(result=observation, reward=reward, done=True)
        finally:
            _rmtree(output_dir)
            self._grading_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Container output extraction (host-side grading input)
    # ------------------------------------------------------------------
    def _extract_output(self, container_path: str = "/app/output") -> str | None:
        """Copy the agent's output tree out of the container to a host temp dir. Returns the host dir
        that directly contains the output files (e.g. warehouse.db), or None if it doesn't exist.
        Needs the Docker backend (it reads the container via the Docker SDK): with any other backend
        there is nothing to grade, so Φ stays 0 and only the terminal reward remains."""
        backend = self._backend
        container = getattr(backend, "_container", None)
        if container is None:
            return None
        try:
            stream, _stat = container.get_archive(container_path)
        except Exception:
            return None  # output/ not created yet
        tmp = tempfile.mkdtemp(prefix="v72-out-")
        try:
            tar_bytes = b"".join(stream)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
                _safe_extractall(tar, tmp)
        except Exception as e:
            logger.warning("[%s] failed to extract %s: %s", self._task_id, container_path, e)
            _rmtree(tmp)
            return None
        # get_archive("/app/output") yields a top-level "output/" dir inside the tar.
        base = os.path.basename(container_path.rstrip("/"))
        extracted = os.path.join(tmp, base)
        return extracted if os.path.isdir(extracted) else tmp

    def get_metrics(self) -> dict[str, float]:
        metrics = super().get_metrics()
        metrics["progress_potential_final"] = float(self._prev_potential)
        metrics["terminal_reward_raw"] = float(self._terminal_raw)
        metrics["terminal_bool_mean"] = float(self._terminal_bool_mean)
        # Host-side grading cost per rollout (Φ evaluations actually run, excl. unchanged-tree cache hits;
        # wall seconds in output extraction + Φ + terminal grading).
        metrics["progress_grade_calls"] = float(self._progress_grade_calls)
        metrics["grading_seconds"] = float(self._grading_seconds)
        return metrics

    def _teardown_grader(self) -> None:
        if self._grader is not None:
            try:
                self._grader.close()
            except Exception:
                pass
            self._grader = None

    async def close(self) -> None:
        self._teardown_grader()
        await super().close()


def _grader_error_is_fatal(exc: BaseException) -> bool:
    """A V72Grader-init failure is fatal (genuine misconfig) ONLY if v7.2 is not importable at all.

    Everything else — a node churn that wiped the local venv/interpreter, an FS race, a
    package-validation "runtime dependency discovery" miss (e.g. FileNotFoundError on the venv's
    python) — is recoverable: degrade the rollout, keep training. A false
    "fatal" kills the whole distributed job (turning a single node preemption into a failed run);
    a false "transient" only skips grading one rollout (visible via progress_potential_final==0). So we
    fatal only on the unambiguous "v7.2 not deployed" case: ImportError/ModuleNotFoundError anywhere in
    the cause chain.
    """
    seen: BaseException | None = exc
    for _ in range(8):
        if isinstance(seen, (ImportError, ModuleNotFoundError)):
            return True
        seen = seen.__cause__ or seen.__context__
        if seen is None:
            break
    return False


def _tree_hash(root: str) -> str:
    """Content hash of a directory tree (relative paths + file bytes). Order-independent."""
    h = hashlib.sha256()
    entries = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            entries.append((os.path.relpath(full, root), full))
    for rel, full in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            with open(full, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def _safe_extractall(tar: tarfile.TarFile, dest: str) -> None:
    """Extract a tar guarding against path traversal (agent-authored file names)."""
    dest_abs = os.path.abspath(dest)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            continue  # skip members that would escape dest
        tar.extract(member, dest)


def _rmtree(path: str | None) -> None:
    if not path:
        return
    import shutil

    shutil.rmtree(path, ignore_errors=True)


@dataclass
class SWERLVanilluxDenseSandboxEnvConfig(BaseEnvConfig):
    """Configuration for SWERLVanilluxDenseSandboxEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = SWERLVanilluxDenseSandboxEnv
    backend: str = "docker"
    image: str = "python:3.12-slim"
    mem_limit: str = "4g"
    penalty: float = -0.05
    task_data_dir: str = ""
    task_data_hf_repo: str = ""
    test_timeout: int = 600
    timeout: int = 120
    last_step_warning: bool = False
    # Dense-reward knobs:
    grade_mode: str = "live"  # default for a task lacking dense_config.json
    require_grader: bool = False  # raise at reset (instead of silent terminal-only) if the grader can't init
    shaping_gamma: float = 1.0
    shaping_beta: float | None = None
    v72_codebase_path: str | None = None
    authority_subdir: str = "authority/package"
