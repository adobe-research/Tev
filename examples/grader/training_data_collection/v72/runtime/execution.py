"""Shared candidate/final workspace, output seal, and verifier boundary."""

from __future__ import annotations

import ctypes
import importlib
import inspect
import shutil
import socket
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from training_data_collection.common.filesystem import (
    file_sha256,
    make_tree_read_only,
    tree_manifest,
    tree_sha256,
)
from training_data_collection.common.identity import canonical_json, content_digest
from training_data_collection.v72.contracts.public_task import (
    PublicTaskSpecification,
    TaskPackageManifest,
)
from training_data_collection.v72.contracts.qualification import PolicyDigestSet
from training_data_collection.v72.contracts.reward import RewardContract, RewardResult
from training_data_collection.v72.contracts.runtime import (
    REQUIRED_RUNTIME_CHECKS,
    REQUIRED_RUNTIME_ENVIRONMENT_VARIABLES,
    BaselinePolicyIdentity,
    ExecutionProfile,
    ProductionProfileQualification,
    RuntimeAccessDenial,
    RuntimeBoundaryAuthority,
    RuntimeBoundaryProbeEvidence,
    RuntimeCheckEvidence,
    RuntimeDryRunEvidence,
    RuntimeStatusSeparationEvidence,
    SealedOutputFile,
    SealedOutputManifest,
    VerifierBundleAuthority,
)
from training_data_collection.v72.qualification.independence import build_dependency_closure
from training_data_collection.v72.runtime.process import (
    JsonProcessExecution,
    JsonProcessRequest,
    execute_json_process,
)
from training_data_collection.v72.runtime.workspace import (
    PublicWorkspaceManifest,
    build_public_workspace,
    validate_public_workspace,
)

ExecutionMode = Literal["candidate", "final"]
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PROCESS_RUNNER_ENTRYPOINT = "training_data_collection.v72.runtime.process:execute_json_process"


def _runtime_environment(code_root: Path, profile: ExecutionProfile) -> dict[str, str]:
    if profile.environment_variable_names != REQUIRED_RUNTIME_ENVIRONMENT_VARIABLES:
        raise ValueError("execution profile has an unsafe runtime environment")
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(code_root),
    }


@dataclass(frozen=True)
class RuntimeQualificationTask:
    """Real package, solver, verifier, and private roots used for runtime qualification."""

    task_manifest: TaskPackageManifest
    specification: PublicTaskSpecification
    workspace_manifest: PublicWorkspaceManifest
    public_input_root: Path
    verifier_bundle: VerifierBundleAuthority
    reward_contract: RewardContract
    verifier_private_root: Path


def _entrypoint_source_sha256(entrypoint: str) -> str:
    try:
        module_name, qualname = entrypoint.split(":", 1)
        target: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        source_path = inspect.getsourcefile(target)
    except (AttributeError, ImportError, ValueError) as error:
        raise ValueError(f"runtime entrypoint is not importable: {entrypoint}") from error
    if not callable(target) or source_path is None:
        raise ValueError(f"runtime entrypoint is not a source-backed callable: {entrypoint}")
    return file_sha256(Path(source_path))


def runtime_process_runner_sha256() -> str:
    """Return the exact dependency-closure digest of the generic sandbox runner."""
    return build_dependency_closure(
        _PROCESS_RUNNER_ENTRYPOINT,
        _SOURCE_ROOT,
    ).dependency_closure_sha256


def run_runtime_boundary_probe(payload: dict[str, object]) -> dict[str, object]:
    """Attempt hostile direct file reads and a real network connection."""
    if set(payload) != {"private_targets"}:
        raise ValueError("runtime boundary probe payload is invalid")
    private_targets = payload["private_targets"]
    if type(private_targets) is not list or not all(
        type(item) is dict
        and set(item) == {"label", "path"}
        and type(item["label"]) is str
        and type(item["path"]) is str
        for item in private_targets
    ):
        raise ValueError("runtime boundary probe payload types are invalid")
    attempts = []
    targets = [*private_targets, {"label": "host-filesystem", "path": "/etc/passwd"}]
    for target in targets:
        try:
            Path(target["path"]).read_bytes()
        except OSError as error:
            attempts.append(
                {
                    "label": target["label"],
                    "status": "denied",
                    "error": type(error).__name__,
                }
            )
        else:
            attempts.append({"label": target["label"], "status": "allowed", "error": None})
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.25):
            network_status = "allowed"
    except OSError:
        network_status = "denied"
    mount_result = ctypes.CDLL(None, use_errno=True).mount(b"tmpfs", b"/tmp", b"tmpfs", 0, None)
    capability_status = "denied" if mount_result != 0 else "allowed"
    return {
        "attempts": attempts,
        "network_status": network_status,
        "capability_status": capability_status,
    }


def run_runtime_status_probe(payload: dict[str, object]) -> dict[str, object]:
    """Produce controlled runtime status outcomes."""
    if set(payload) != {"mode"} or payload["mode"] not in {
        "success",
        "failure",
        "resource",
        "timeout",
    }:
        raise ValueError("runtime status probe payload is invalid")
    if payload["mode"] == "failure":
        raise RuntimeError("controlled policy failure")
    if payload["mode"] == "timeout":
        time.sleep(60.0)
    if payload["mode"] == "resource":
        Path("output/resource-a.bin").write_bytes(b"a")
        Path("output/resource-b.bin").write_bytes(b"b")
    return {"status": "success"}


def stage_entrypoint_closure(
    entrypoint: str,
    destination: Path,
    *,
    source_root: Path = _SOURCE_ROOT,
) -> Path:
    """Stage the exact audited dependency closure for one process entrypoint."""
    closure = build_dependency_closure(entrypoint, source_root)
    destination.mkdir(parents=True)
    for record in closure.modules:
        source = source_root / record.relative_path
        target = destination / record.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        relative = Path(record.relative_path)
        for parent in relative.parents:
            if parent == Path("."):
                continue
            source_init = source_root / parent / "__init__.py"
            target_init = destination / parent / "__init__.py"
            if source_init.is_file() and not target_init.exists():
                shutil.copy2(source_init, target_init)
    return destination


def sandbox_read_roots(*roots: Path) -> tuple[Path, ...]:
    """Return the exact host runtime roots required by the chroot process boundary."""
    candidates = [
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/dev/null"),
        Path("/dev/random"),
        Path("/dev/urandom"),
        Path(sys.prefix),
        Path(sys.prefix).resolve(strict=True),
        Path(sys.executable).resolve(strict=True).parents[1],
        *roots,
    ]
    return tuple(sorted({path.resolve(strict=True) for path in candidates if path.exists()}))


def _run_runtime_mode(
    *,
    mode: ExecutionMode,
    root: Path,
    task: RuntimeQualificationTask,
    baseline_policy: BaselinePolicyIdentity,
    execution_profile: ExecutionProfile,
) -> tuple[
    PublicWorkspaceManifest,
    SealedOutputManifest,
    JsonProcessExecution,
    JsonProcessExecution,
    RewardResult,
]:
    workspace_root = root / f"{mode}-workspace"
    manifest = build_runtime_workspace(
        mode=mode,
        specification=task.specification,
        input_source=task.public_input_root,
        destination=workspace_root,
        expected_manifest=task.workspace_manifest,
    )
    (workspace_root / "output").rmdir()
    (workspace_root / "output").mkdir()
    policy_code_root = stage_entrypoint_closure(
        baseline_policy.policy_entrypoint,
        root / f"{mode}-policy-code",
    )
    if tree_sha256(policy_code_root) != baseline_policy.policy_code_closure_sha256:
        raise ValueError("runtime baseline policy staged closure is stale")
    environment = _runtime_environment(policy_code_root, execution_profile)
    policy_request = JsonProcessRequest.identified(
        entrypoint=baseline_policy.policy_entrypoint,
        payload={
            "public_input_root": str(workspace_root / "input"),
            "output_root": str(workspace_root / "output"),
        },
        authority_context={
            "role": "policy",
            "mode": mode,
            "task_id": task.task_manifest.task_id,
            "workspace_manifest": manifest.model_dump(mode="json"),
            "baseline_policy": baseline_policy.model_dump(mode="json"),
        },
        cwd=workspace_root,
        environment=environment,
        timeout_seconds=execution_profile.resource_limits.maximum_elapsed_seconds,
        isolation="policy",
        read_only_roots=sandbox_read_roots(policy_code_root, workspace_root),
        writable_roots=(workspace_root / "output",),
        maximum_cpu_seconds=execution_profile.resource_limits.maximum_cpu_seconds,
        maximum_memory_bytes=execution_profile.resource_limits.maximum_peak_memory_bytes,
        maximum_file_bytes=execution_profile.resource_limits.maximum_disk_bytes,
        maximum_output_entries=execution_profile.resource_limits.maximum_output_entries,
        code_root=policy_code_root,
    )
    policy_execution = execute_json_process(policy_request)
    if policy_execution.status != "success" or type(policy_execution.response) is not dict:
        raise ValueError(
            f"{mode} policy runtime did not complete successfully: "
            f"{policy_execution.error_detail}; {policy_execution.stderr}"
        )
    output_sha256 = tree_sha256(workspace_root / "output")
    if policy_execution.response.get("output_tree_sha256") != output_sha256:
        raise ValueError(f"{mode} policy process output digest is stale")
    sealed = seal_policy_output(
        workspace_root=workspace_root,
        specification=task.specification,
        workspace_manifest=manifest,
        task_id=task.task_manifest.task_id,
    )
    private_root = task.verifier_private_root.resolve(strict=True)
    if tree_sha256(private_root) != task.verifier_bundle.private_tree_sha256:
        raise ValueError("runtime verifier private tree differs from bundle authority")
    verifier_code_root = stage_entrypoint_closure(
        task.verifier_bundle.verifier_entrypoint,
        root / f"{mode}-verifier-code",
    )
    if tree_sha256(verifier_code_root) != task.verifier_bundle.verifier_code_closure_sha256:
        raise ValueError("runtime verifier staged closure is stale")
    verifier_environment = _runtime_environment(verifier_code_root, execution_profile)
    verifier_request = JsonProcessRequest.identified(
        entrypoint=task.verifier_bundle.verifier_entrypoint,
        payload={
            "task_id": task.task_manifest.task_id,
            "reward_contract": task.reward_contract.model_dump(mode="json"),
            "public_input_root": str(workspace_root / "input"),
            "sealed_output_root": str(workspace_root / "output"),
        },
        authority_context={
            "role": "verifier",
            "mode": mode,
            "task_id": task.task_manifest.task_id,
            "workspace_manifest": manifest.model_dump(mode="json"),
            "sealed_output": sealed.model_dump(mode="json"),
            "verifier_bundle": task.verifier_bundle.model_dump(mode="json"),
            "reward_contract": task.reward_contract.model_dump(mode="json"),
        },
        cwd=workspace_root,
        environment=verifier_environment,
        timeout_seconds=execution_profile.resource_limits.maximum_elapsed_seconds,
        isolation="verifier",
        read_only_roots=sandbox_read_roots(
            verifier_code_root,
            workspace_root,
            private_root,
        ),
        maximum_cpu_seconds=execution_profile.resource_limits.maximum_cpu_seconds,
        maximum_memory_bytes=execution_profile.resource_limits.maximum_peak_memory_bytes,
        maximum_file_bytes=execution_profile.resource_limits.maximum_disk_bytes,
        maximum_output_entries=execution_profile.resource_limits.maximum_output_entries,
        code_root=verifier_code_root,
    )
    verifier_execution = execute_json_process(verifier_request)
    if verifier_execution.status != "success" or type(verifier_execution.response) is not dict:
        raise ValueError(
            f"{mode} verifier runtime did not complete successfully: "
            f"{verifier_execution.error_detail}; {verifier_execution.stderr}"
        )
    reward = RewardResult.model_validate_json(canonical_json(verifier_execution.response))
    if reward.status != "policy_result" or reward.task_id != task.task_manifest.task_id:
        raise ValueError(f"{mode} verifier did not produce a policy reward")
    validate_sealed_output(workspace_root, sealed)
    return manifest, sealed, policy_execution, verifier_execution, reward


def qualify_production_profile(
    *,
    execution_profile: ExecutionProfile,
    runtime_boundary: RuntimeBoundaryAuthority,
    baseline_policy: BaselinePolicyIdentity,
    policy_digest_set: PolicyDigestSet,
    tasks: tuple[RuntimeQualificationTask, ...],
) -> ProductionProfileQualification:
    """Execute candidate/final two-root parity and every required runtime check."""
    if not tasks:
        raise ValueError("production runtime qualification requires at least one task")
    if runtime_boundary != current_runtime_boundary_authority(
        private_path_markers=runtime_boundary.private_path_markers
    ):
        raise ValueError("runtime boundary differs from the current implementation authority")
    runner_sha256 = runtime_process_runner_sha256()
    if (
        execution_profile.policy_runner_sha256 != runner_sha256
        or execution_profile.verifier_runner_sha256 != runner_sha256
    ):
        raise ValueError("production profile differs from the generic sandbox process runner")
    if baseline_policy.policy_entrypoint_sha256 != _entrypoint_source_sha256(
        baseline_policy.policy_entrypoint
    ):
        raise ValueError("production baseline policy entrypoint authority is stale")
    ordered_tasks = tuple(sorted(tasks, key=lambda item: item.task_manifest.task_id))
    if len({task.task_manifest.task_id for task in ordered_tasks}) != len(ordered_tasks):
        raise ValueError("production runtime qualification tasks must be unique")
    dry_runs: list[RuntimeDryRunEvidence] = []
    process_executions: list[JsonProcessExecution] = []
    with tempfile.TemporaryDirectory(prefix="v72-runtime-qualification-") as raw:
        qualification_root = Path(raw)
        for index, task in enumerate(ordered_tasks):
            if (
                task.verifier_bundle.verifier_entrypoint_sha256
                != _entrypoint_source_sha256(task.verifier_bundle.verifier_entrypoint)
                or task.reward_contract.contract_id != task.verifier_bundle.reward_contract_id
                or task.workspace_manifest.input_tree_sha256
                != task.task_manifest.public_input_tree_sha256
            ):
                raise ValueError("runtime task entrypoint or package authority is stale")
            task_root = qualification_root / f"task-{index}"
            task_root.mkdir()
            candidate = _run_runtime_mode(
                mode="candidate",
                root=task_root,
                task=task,
                baseline_policy=baseline_policy,
                execution_profile=execution_profile,
            )
            final = _run_runtime_mode(
                mode="final",
                root=task_root,
                task=task,
                baseline_policy=baseline_policy,
                execution_profile=execution_profile,
            )
            (
                candidate_manifest,
                candidate_seal,
                candidate_policy,
                candidate_verifier,
                candidate_reward,
            ) = candidate
            final_manifest, final_seal, final_policy, final_verifier, final_reward = final
            process_executions.extend(
                (
                    candidate_policy,
                    candidate_verifier,
                    final_policy,
                    final_verifier,
                )
            )
            dry_runs.append(
                RuntimeDryRunEvidence.identified(
                    execution_profile_id=execution_profile.profile_id,
                    runtime_boundary_id=runtime_boundary.runtime_boundary_id,
                    task_id=task.task_manifest.task_id,
                    candidate_workspace_id=candidate_manifest.workspace_id,
                    final_workspace_id=final_manifest.workspace_id,
                    candidate_policy_execution_id=candidate_policy.execution_id,
                    final_policy_execution_id=final_policy.execution_id,
                    candidate_verifier_execution_id=candidate_verifier.execution_id,
                    final_verifier_execution_id=final_verifier.execution_id,
                    candidate_output_sha256=candidate_seal.output_tree_sha256,
                    final_output_sha256=final_seal.output_tree_sha256,
                    candidate_reward_sha256=content_digest(
                        candidate_reward.model_dump(mode="json")
                    ),
                    final_reward_sha256=content_digest(final_reward.model_dump(mode="json")),
                )
            )
        probe_root = qualification_root / "boundary-probe"
        probe_root.mkdir()
        (probe_root / "output").mkdir()
        probe_code_root = stage_entrypoint_closure(
            "training_data_collection.v72.runtime.execution:run_runtime_boundary_probe",
            qualification_root / "probe-code",
        )
        private_paths = tuple(
            str(task.verifier_private_root.resolve(strict=True)) for task in ordered_tasks
        )
        probe = execute_json_process(
            JsonProcessRequest.identified(
                entrypoint=(
                    "training_data_collection.v72.runtime.execution:run_runtime_boundary_probe"
                ),
                payload={
                    "private_targets": [
                        {"label": f"verifier-private-{index}", "path": path}
                        for index, path in enumerate(private_paths)
                    ],
                },
                authority_context={
                    "role": "boundary_probe",
                    "execution_profile_id": execution_profile.profile_id,
                    "runtime_boundary_id": runtime_boundary.runtime_boundary_id,
                },
                cwd=probe_root,
                environment=_runtime_environment(probe_code_root, execution_profile),
                timeout_seconds=min(
                    30.0,
                    execution_profile.resource_limits.maximum_elapsed_seconds,
                ),
                isolation="policy",
                read_only_roots=sandbox_read_roots(probe_code_root, probe_root),
                writable_roots=(probe_root / "output",),
                maximum_cpu_seconds=execution_profile.resource_limits.maximum_cpu_seconds,
                maximum_memory_bytes=execution_profile.resource_limits.maximum_peak_memory_bytes,
                maximum_file_bytes=execution_profile.resource_limits.maximum_disk_bytes,
                maximum_output_entries=(execution_profile.resource_limits.maximum_output_entries),
                code_root=probe_code_root,
            )
        )
        if (
            probe.status != "success"
            or type(probe.response) is not dict
            or any(item.get("status") != "denied" for item in probe.response.get("attempts", []))
            or len(probe.response.get("attempts", [])) != len(private_paths) + 1
            or probe.response.get("network_status") != "denied"
            or probe.response.get("capability_status") != "denied"
        ):
            raise ValueError("runtime private access-denial probe did not execute completely")
        process_executions.append(probe)
        access_denials = tuple(
            sorted(
                (
                    RuntimeAccessDenial(
                        target_sha256=content_digest({"hostile_access_target": attempt["label"]}),
                        target_kind=(
                            "host_filesystem"
                            if attempt["label"] == "host-filesystem"
                            else "verifier_private"
                        ),
                        outcome="denied",
                        error_kind=attempt["error"],
                    )
                    for attempt in probe.response["attempts"]
                ),
                key=lambda item: item.target_sha256,
            )
        )
        boundary_probe_evidence = RuntimeBoundaryProbeEvidence.identified(
            execution_profile_id=execution_profile.profile_id,
            runtime_boundary_id=runtime_boundary.runtime_boundary_id,
            process_execution=probe,
            policy_code_closure_sha256=tree_sha256(probe_code_root),
            access_denials=access_denials,
            network_outcome="denied",
            mount_outcome="denied",
        )

        status_root = qualification_root / "status-probe"
        status_root.mkdir()
        (status_root / "output").mkdir()
        status_code_root = stage_entrypoint_closure(
            "training_data_collection.v72.runtime.execution:run_runtime_status_probe",
            qualification_root / "status-code",
        )
        status_environment = _runtime_environment(status_code_root, execution_profile)

        def execute_status_probe(
            mode: str, *, entrypoint: str | None = None
        ) -> JsonProcessExecution:
            return execute_json_process(
                JsonProcessRequest.identified(
                    entrypoint=entrypoint
                    or "training_data_collection.v72.runtime.execution:run_runtime_status_probe",
                    payload={"mode": mode},
                    authority_context={
                        "role": "status_probe",
                        "expected_status": (
                            "infrastructure_failure"
                            if entrypoint is not None
                            else {
                                "success": "success",
                                "failure": "policy_failure",
                                "resource": "resource_failure",
                                "timeout": "timeout",
                            }[mode]
                        ),
                    },
                    cwd=status_root,
                    environment=status_environment,
                    timeout_seconds=(
                        min(0.05, execution_profile.resource_limits.maximum_elapsed_seconds)
                        if mode == "timeout"
                        else min(30.0, execution_profile.resource_limits.maximum_elapsed_seconds)
                    ),
                    isolation="policy",
                    read_only_roots=sandbox_read_roots(status_code_root, status_root),
                    writable_roots=(status_root / "output",),
                    maximum_cpu_seconds=execution_profile.resource_limits.maximum_cpu_seconds,
                    maximum_memory_bytes=(
                        execution_profile.resource_limits.maximum_peak_memory_bytes
                    ),
                    maximum_file_bytes=(
                        1
                        if mode == "resource"
                        else execution_profile.resource_limits.maximum_disk_bytes
                    ),
                    maximum_output_entries=(
                        execution_profile.resource_limits.maximum_output_entries
                    ),
                    code_root=status_code_root,
                )
            )

        success_execution = execute_status_probe("success")
        policy_failure_execution = execute_status_probe("failure")
        resource_failure_execution = execute_status_probe("resource")
        timeout_execution = execute_status_probe("timeout")
        infrastructure_failure_execution = execute_status_probe(
            "success",
            entrypoint="training_data_collection.v72.runtime.missing:run",
        )
        status_executions = (
            success_execution,
            policy_failure_execution,
            resource_failure_execution,
            timeout_execution,
            infrastructure_failure_execution,
        )
        if tuple(item.status for item in status_executions) != (
            "success",
            "policy_failure",
            "resource_failure",
            "timeout",
            "infrastructure_failure",
        ):
            raise ValueError(
                "runtime status probe failed to separate process outcomes: "
                f"{tuple(item.status for item in status_executions)}"
            )
        process_executions.extend(status_executions)
        status_separation_evidence = RuntimeStatusSeparationEvidence.identified(
            execution_profile_id=execution_profile.profile_id,
            runtime_boundary_id=runtime_boundary.runtime_boundary_id,
            success_execution=success_execution,
            policy_failure_execution=policy_failure_execution,
            resource_failure_execution=resource_failure_execution,
            timeout_execution=timeout_execution,
            infrastructure_failure_execution=infrastructure_failure_execution,
        )

    ordered_executions = tuple(sorted(process_executions, key=lambda item: item.execution_id))
    limits = execution_profile.resource_limits
    if any(
        item.elapsed_seconds > limits.maximum_elapsed_seconds
        or item.cpu_seconds > limits.maximum_cpu_seconds
        or item.peak_memory_bytes > limits.maximum_peak_memory_bytes
        or item.output_bytes > limits.maximum_disk_bytes
        or item.output_entries > limits.maximum_output_entries
        for item in ordered_executions
    ):
        raise ValueError("executed runtime resource measurement exceeds the selected profile")
    policy_ids = tuple(
        sorted(
            execution_id
            for evidence in dry_runs
            for execution_id in (
                evidence.candidate_policy_execution_id,
                evidence.final_policy_execution_id,
            )
        )
    )
    verifier_ids = tuple(
        sorted(
            execution_id
            for evidence in dry_runs
            for execution_id in (
                evidence.candidate_verifier_execution_id,
                evidence.final_verifier_execution_id,
            )
        )
    )
    dry_run_execution_ids = tuple(sorted((*policy_ids, *verifier_ids)))
    boundary_id = boundary_probe_evidence.process_execution.execution_id
    status_ids = tuple(sorted(item.execution_id for item in status_executions))
    all_execution_ids = tuple(item.execution_id for item in ordered_executions)
    executions_by_id = {item.execution_id: item for item in ordered_executions}
    check_authority = {
        "workspace_determinism": (
            dry_run_execution_ids,
            content_digest([item.model_dump(mode="json") for item in dry_runs]),
        ),
        "policy_private_isolation": (
            (boundary_id,),
            content_digest(boundary_probe_evidence.model_dump(mode="json")),
        ),
        "output_sealing": (
            policy_ids,
            content_digest(
                {
                    "dry_runs": [item.model_dump(mode="json") for item in dry_runs],
                    "policy_executions": [
                        executions_by_id[item].model_dump(mode="json") for item in policy_ids
                    ],
                }
            ),
        ),
        "verifier_private_isolation": (
            tuple(sorted((*verifier_ids, boundary_id))),
            content_digest(
                {
                    "boundary_probe": boundary_probe_evidence.model_dump(mode="json"),
                    "verifier_executions": [
                        executions_by_id[item].model_dump(mode="json") for item in verifier_ids
                    ],
                }
            ),
        ),
        "status_separation": (
            status_ids,
            content_digest(status_separation_evidence.model_dump(mode="json")),
        ),
        "measured_resource_cost": (
            all_execution_ids,
            content_digest(
                {
                    "resource_limits": limits.model_dump(mode="json"),
                    "process_executions": [
                        item.model_dump(mode="json") for item in ordered_executions
                    ],
                }
            ),
        ),
    }
    checks = tuple(
        RuntimeCheckEvidence.identified(
            execution_profile_id=execution_profile.profile_id,
            runtime_boundary_id=runtime_boundary.runtime_boundary_id,
            check_kind=check_kind,
            evidence_sha256=check_authority[check_kind][1],
            execution_evidence_ids=check_authority[check_kind][0],
            result="satisfied",
        )
        for check_kind in REQUIRED_RUNTIME_CHECKS
    )
    return ProductionProfileQualification.identified(
        execution_profile=execution_profile,
        runtime_boundary=runtime_boundary,
        baseline_policy=baseline_policy,
        policy_digest_set=policy_digest_set,
        dry_run_evidence=tuple(dry_runs),
        boundary_probe_evidence=boundary_probe_evidence,
        status_separation_evidence=status_separation_evidence,
        process_executions=ordered_executions,
        runtime_checks=checks,
    )


def build_runtime_workspace(
    *,
    mode: ExecutionMode,
    specification: PublicTaskSpecification,
    input_source: Path,
    destination: Path,
    expected_manifest: PublicWorkspaceManifest,
) -> PublicWorkspaceManifest:
    """Build the same exact workspace for candidate qualification and final execution."""
    if mode not in {"candidate", "final"}:
        raise ValueError("runtime mode must be candidate or final")
    specification = PublicTaskSpecification.model_validate(specification.model_dump(mode="python"))
    expected_manifest = PublicWorkspaceManifest.model_validate(
        expected_manifest.model_dump(mode="python")
    )
    software_source = input_source.parent / "software"
    manifest = build_public_workspace(
        specification,
        input_source,
        destination,
        software_source=software_source if software_source.is_dir() else None,
    )
    if manifest != expected_manifest:
        raise ValueError("runtime workspace differs from candidate package authority")
    validate_public_workspace(destination, specification, manifest)
    return manifest


def validate_sealed_output(
    workspace_root: Path,
    sealed_output: SealedOutputManifest,
) -> None:
    """Recompute a sealed output tree and require every path to remain immutable."""
    workspace = workspace_root.resolve(strict=True)
    output_root = workspace / "output"
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("sealed output root must be a real directory")
    records = tree_manifest(output_root)
    actual_files = tuple(
        SealedOutputFile(
            path=str(record["path"]),
            sha256=str(record["sha256"]),
            size_bytes=int(record["size_bytes"]),
        )
        for record in records
    )
    if actual_files != sealed_output.files or tree_sha256(output_root) != (
        sealed_output.output_tree_sha256
    ):
        raise ValueError("policy output differs from sealed output authority")
    if sealed_output.sealed_output_id != sealed_output.expected_sealed_output_id():
        raise ValueError("sealed output identity is stale")
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if any(path.stat().st_mode & write_bits for path in (output_root, *output_root.rglob("*"))):
        raise ValueError("sealed output tree contains a writable path")


def seal_policy_output(
    *,
    workspace_root: Path,
    specification: PublicTaskSpecification,
    workspace_manifest: PublicWorkspaceManifest,
    task_id: str,
) -> SealedOutputManifest:
    """Freeze policy output and return its content identity before verifier startup."""
    validate_public_workspace(workspace_root, specification, workspace_manifest)
    output_root = workspace_root.resolve(strict=True) / "output"
    records = tree_manifest(output_root)
    declared_paths = {artifact.path for artifact in specification.output_artifacts}
    actual_paths = {str(record["path"]) for record in records}
    if not declared_paths.issubset(actual_paths):
        raise ValueError("policy output is missing a declared artifact")
    if not specification.public_evaluation.allow_extra_files and actual_paths != declared_paths:
        raise ValueError("policy output contains an undeclared artifact")
    if not specification.public_evaluation.allow_extra_files:
        expected_directories = {
            parent.as_posix()
            for path in declared_paths
            for parent in Path(path).parents
            if parent != Path(".")
        }
        actual_directories = {
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*")
            if path.is_dir()
        }
        if actual_directories != expected_directories:
            raise ValueError("policy output contains an undeclared directory")
    files = tuple(
        SealedOutputFile(
            path=str(record["path"]),
            sha256=str(record["sha256"]),
            size_bytes=int(record["size_bytes"]),
        )
        for record in records
    )
    manifest = SealedOutputManifest.identified(
        task_id=task_id,
        public_workspace_id=workspace_manifest.workspace_id,
        output_tree_sha256=tree_sha256(output_root),
        files=files,
    )
    make_tree_read_only(output_root)
    validate_sealed_output(workspace_root, manifest)
    validate_public_workspace(workspace_root, specification, workspace_manifest)
    return manifest


def current_runtime_boundary_authority(
    *, private_path_markers: tuple[str, ...]
) -> RuntimeBoundaryAuthority:
    """Derive the exact current workspace, runner, and output-sealer authority."""
    execution_closure_sha256 = build_dependency_closure(
        "training_data_collection.v72.runtime.execution:build_runtime_workspace",
        _SOURCE_ROOT,
    ).dependency_closure_sha256
    sealer_closure_sha256 = build_dependency_closure(
        "training_data_collection.v72.runtime.execution:seal_policy_output",
        _SOURCE_ROOT,
    ).dependency_closure_sha256
    runner_sha256 = runtime_process_runner_sha256()
    return RuntimeBoundaryAuthority.identified(
        workspace_builder_sha256=execution_closure_sha256,
        policy_sandbox_sha256=runner_sha256,
        output_sealer_sha256=sealer_closure_sha256,
        verifier_sandbox_sha256=runner_sha256,
        private_path_markers=private_path_markers,
    )


__all__ = [
    "ExecutionMode",
    "RuntimeQualificationTask",
    "build_runtime_workspace",
    "current_runtime_boundary_authority",
    "qualify_production_profile",
    "run_runtime_boundary_probe",
    "run_runtime_status_probe",
    "runtime_process_runner_sha256",
    "sandbox_read_roots",
    "seal_policy_output",
    "stage_entrypoint_closure",
    "validate_sealed_output",
]
