"""Content-bound production runtime and sealed-output authority."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, ValidationInfo, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import canonical_json, content_digest, content_id
from training_data_collection.v72.runtime.process import JsonProcessExecution
from training_data_collection.v72.runtime.workspace import PublicWorkspaceManifest

from .identity import ContentIdentifier, Sha256Digest, StrictFrozenModel
from .qualification import PolicyDigestSet
from .reward import RewardContract, RewardResult

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
PositiveInteger = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]

PUBLIC_RUNTIME_PATHS = ("task_spec.json", "prompt.md", "input", "output")
REQUIRED_RUNTIME_CHECKS = (
    "measured_resource_cost",
    "output_sealing",
    "policy_private_isolation",
    "status_separation",
    "verifier_private_isolation",
    "workspace_determinism",
)
REQUIRED_RUNTIME_ENVIRONMENT_VARIABLES = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONPATH",
)


class ExecutionResourceLimits(StrictFrozenModel):
    """Hard limits enforced by one execution profile."""

    maximum_cpu_seconds: PositiveFloat
    maximum_peak_memory_bytes: PositiveInteger
    maximum_disk_bytes: PositiveInteger
    maximum_output_entries: PositiveInteger
    maximum_elapsed_seconds: PositiveFloat


class ExecutionProfile(StrictFrozenModel):
    """Exact provider, sandbox, runner, and resource authority for rollouts."""

    schema_version: Literal["v7.2.execution-profile.v1"] = "v7.2.execution-profile.v1"
    profile_id: ContentIdentifier
    backend_kind: Literal["local", "container", "provider"]
    provider_name: NonEmptyString
    provider_revision_sha256: Sha256Digest
    sandbox_image_sha256: Sha256Digest
    toolchain_tree_sha256: Sha256Digest
    policy_runner_sha256: Sha256Digest
    verifier_runner_sha256: Sha256Digest
    cpu_architecture: NonEmptyString
    network_access: Literal[False] = False
    environment_variable_names: tuple[NonEmptyString, ...]
    resource_limits: ExecutionResourceLimits

    @model_validator(mode="after")
    def validate_profile(self, info: ValidationInfo) -> Self:
        if self.environment_variable_names != REQUIRED_RUNTIME_ENVIRONMENT_VARIABLES:
            raise ValueError("execution profile must use the exact safe runtime environment")
        if not (info.context or {}).get("skip_identity") and (
            self.profile_id != self.expected_profile_id()
        ):
            raise ValueError("profile_id does not match execution profile authority")
        return self

    def expected_profile_id(self) -> str:
        return content_id(
            "v72-execution-profile",
            self.model_dump(mode="json", exclude={"profile_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"profile_id": f"v72-execution-profile:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            profile_id=provisional.expected_profile_id(),
            **provisional.model_dump(mode="python", exclude={"profile_id"}),
        )


class BaselinePolicyIdentity(StrictFrozenModel):
    """Fixed model, inference, staged code, runner, and compute identity."""

    schema_version: Literal["v7.2.baseline-policy.v1"] = "v7.2.baseline-policy.v1"
    baseline_policy_id: ContentIdentifier
    model_identifier: NonEmptyString
    policy_entrypoint: NonEmptyString
    model_weights_sha256: Sha256Digest
    inference_configuration_sha256: Sha256Digest
    policy_entrypoint_sha256: Sha256Digest
    policy_code_closure_sha256: Sha256Digest
    compute_configuration_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self, info: ValidationInfo) -> Self:
        if not (info.context or {}).get("skip_identity") and (
            self.baseline_policy_id != self.expected_baseline_policy_id()
        ):
            raise ValueError("baseline policy identity is stale")
        return self

    def expected_baseline_policy_id(self) -> str:
        return content_id(
            "v72-baseline-policy",
            self.model_dump(mode="json", exclude={"baseline_policy_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"baseline_policy_id": f"v72-baseline-policy:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            baseline_policy_id=provisional.expected_baseline_policy_id(),
            **provisional.model_dump(mode="python", exclude={"baseline_policy_id"}),
        )


class RuntimeBoundaryAuthority(StrictFrozenModel):
    """Implementation identity shared by candidate and final execution modes."""

    schema_version: Literal["v7.2.runtime-boundary.v1"] = "v7.2.runtime-boundary.v1"
    runtime_boundary_id: ContentIdentifier
    workspace_builder_sha256: Sha256Digest
    policy_sandbox_sha256: Sha256Digest
    output_sealer_sha256: Sha256Digest
    verifier_sandbox_sha256: Sha256Digest
    public_workspace_paths: tuple[
        Literal["task_spec.json", "prompt.md", "input", "output"], ...
    ] = PUBLIC_RUNTIME_PATHS
    private_path_markers: tuple[NonEmptyString, ...]
    output_sealed_before_verifier: Literal[True] = True

    @model_validator(mode="after")
    def validate_boundary(self, info: ValidationInfo) -> Self:
        if self.public_workspace_paths != PUBLIC_RUNTIME_PATHS:
            raise ValueError("runtime boundary must expose the exact public workspace")
        if not self.private_path_markers or self.private_path_markers != tuple(
            sorted(set(self.private_path_markers))
        ):
            raise ValueError("private path markers must be nonempty, sorted, and unique")
        if set(self.private_path_markers).intersection(PUBLIC_RUNTIME_PATHS):
            raise ValueError("private path markers cannot overlap public workspace paths")
        if not (info.context or {}).get("skip_identity") and (
            self.runtime_boundary_id != self.expected_runtime_boundary_id()
        ):
            raise ValueError("runtime_boundary_id does not match runtime authority")
        return self

    def expected_runtime_boundary_id(self) -> str:
        return content_id(
            "v72-runtime-boundary",
            self.model_dump(mode="json", exclude={"runtime_boundary_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"runtime_boundary_id": f"v72-runtime-boundary:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            runtime_boundary_id=provisional.expected_runtime_boundary_id(),
            **provisional.model_dump(mode="python", exclude={"runtime_boundary_id"}),
        )


class VerifierBundleAuthority(StrictFrozenModel):
    """Private verifier bundle selected for one exact task."""

    schema_version: Literal["v7.2.verifier-bundle.v1"] = "v7.2.verifier-bundle.v1"
    verifier_bundle_id: ContentIdentifier
    task_id: ContentIdentifier
    verifier_entrypoint: NonEmptyString
    verifier_entrypoint_sha256: Sha256Digest
    verifier_code_closure_sha256: Sha256Digest
    private_tree_sha256: Sha256Digest
    hidden_suite_id: ContentIdentifier
    reward_contract_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_bundle(self, info: ValidationInfo) -> Self:
        if not (info.context or {}).get("skip_identity") and (
            self.verifier_bundle_id != self.expected_verifier_bundle_id()
        ):
            raise ValueError("verifier_bundle_id does not match verifier authority")
        return self

    def expected_verifier_bundle_id(self) -> str:
        return content_id(
            "v72-verifier-bundle",
            self.model_dump(mode="json", exclude={"verifier_bundle_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"verifier_bundle_id": f"v72-verifier-bundle:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            verifier_bundle_id=provisional.expected_verifier_bundle_id(),
            **provisional.model_dump(mode="python", exclude={"verifier_bundle_id"}),
        )


RuntimeCheckKind = Literal[
    "workspace_determinism",
    "policy_private_isolation",
    "output_sealing",
    "verifier_private_isolation",
    "status_separation",
    "measured_resource_cost",
]


class RuntimeCheckEvidence(StrictFrozenModel):
    """Executable evidence for one required production-profile runtime property."""

    schema_version: Literal["v7.2.runtime-check.v1"] = "v7.2.runtime-check.v1"
    runtime_check_id: ContentIdentifier
    execution_profile_id: ContentIdentifier
    runtime_boundary_id: ContentIdentifier
    check_kind: RuntimeCheckKind
    evidence_sha256: Sha256Digest
    execution_evidence_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    result: Literal["satisfied"] = "satisfied"

    @model_validator(mode="after")
    def validate_check(self, info: ValidationInfo) -> Self:
        if self.execution_evidence_ids != tuple(sorted(set(self.execution_evidence_ids))):
            raise ValueError("runtime check execution evidence must be sorted and unique")
        if not (info.context or {}).get("skip_identity") and (
            self.runtime_check_id != self.expected_runtime_check_id()
        ):
            raise ValueError("runtime_check_id does not match check evidence")
        return self

    def expected_runtime_check_id(self) -> str:
        return content_id(
            "v72-runtime-check",
            self.model_dump(mode="json", exclude={"runtime_check_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"runtime_check_id": f"v72-runtime-check:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            runtime_check_id=provisional.expected_runtime_check_id(),
            **provisional.model_dump(mode="python", exclude={"runtime_check_id"}),
        )


class RuntimeDryRunEvidence(StrictFrozenModel):
    """Two-root candidate/final process execution and parity authority."""

    schema_version: Literal["v7.2.runtime-dry-run.v1"] = "v7.2.runtime-dry-run.v1"
    runtime_dry_run_id: ContentIdentifier
    execution_profile_id: ContentIdentifier
    runtime_boundary_id: ContentIdentifier
    task_id: ContentIdentifier
    candidate_workspace_id: ContentIdentifier
    final_workspace_id: ContentIdentifier
    candidate_policy_execution_id: ContentIdentifier
    final_policy_execution_id: ContentIdentifier
    candidate_verifier_execution_id: ContentIdentifier
    final_verifier_execution_id: ContentIdentifier
    candidate_output_sha256: Sha256Digest
    final_output_sha256: Sha256Digest
    candidate_reward_sha256: Sha256Digest
    final_reward_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_parity(self, info: ValidationInfo) -> Self:
        if (
            self.candidate_workspace_id != self.final_workspace_id
            or self.candidate_output_sha256 != self.final_output_sha256
            or self.candidate_reward_sha256 != self.final_reward_sha256
            or self.candidate_policy_execution_id == self.final_policy_execution_id
            or self.candidate_verifier_execution_id == self.final_verifier_execution_id
        ):
            raise ValueError("runtime dry run does not prove independent two-root parity")
        if not (info.context or {}).get("skip_identity") and (
            self.runtime_dry_run_id != self.expected_runtime_dry_run_id()
        ):
            raise ValueError("runtime dry-run identity is stale")
        return self

    def expected_runtime_dry_run_id(self) -> str:
        return content_id(
            "v72-runtime-dry-run",
            self.model_dump(mode="json", exclude={"runtime_dry_run_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"runtime_dry_run_id": f"v72-runtime-dry-run:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            runtime_dry_run_id=provisional.expected_runtime_dry_run_id(),
            **provisional.model_dump(mode="python", exclude={"runtime_dry_run_id"}),
        )


class RuntimeAccessDenial(StrictFrozenModel):
    """One hostile filesystem read rejected by the isolated policy boundary."""

    target_sha256: Sha256Digest
    target_kind: Literal["verifier_private", "host_filesystem"]
    outcome: Literal["denied"]
    error_kind: NonEmptyString


class RuntimeBoundaryProbeEvidence(StrictFrozenModel):
    """Executed hostile read, network, and remount denial authority."""

    schema_version: Literal["v7.2.runtime-boundary-probe.v1"] = "v7.2.runtime-boundary-probe.v1"
    runtime_boundary_probe_id: ContentIdentifier
    execution_profile_id: ContentIdentifier
    runtime_boundary_id: ContentIdentifier
    process_execution: JsonProcessExecution
    policy_code_closure_sha256: Sha256Digest
    access_denials: tuple[RuntimeAccessDenial, ...] = Field(min_length=2)
    network_outcome: Literal["denied"]
    mount_outcome: Literal["denied"]

    @model_validator(mode="after")
    def validate_probe(self, info: ValidationInfo) -> Self:
        if self.process_execution.status != "success":
            raise ValueError("runtime boundary probe must embed a successful process execution")
        if (
            self.process_execution.request.authority_context
            != {
                "role": "boundary_probe",
                "execution_profile_id": self.execution_profile_id,
                "runtime_boundary_id": self.runtime_boundary_id,
            }
            or self.process_execution.request.isolation != "policy"
            or self.process_execution.request.code_closure_sha256 != self.policy_code_closure_sha256
        ):
            raise ValueError("runtime boundary probe process authority is incomplete")
        response = self.process_execution.response
        if (
            type(response) is not dict
            or set(response) != {"attempts", "network_status", "capability_status"}
            or type(response["attempts"]) is not list
            or response["network_status"] not in {"allowed", "denied"}
            or response["capability_status"] not in {"allowed", "denied"}
        ):
            raise ValueError("runtime boundary probe response is malformed")
        attempts = response["attempts"]
        if any(
            type(item) is not dict
            or set(item) != {"label", "status", "error"}
            or type(item["label"]) is not str
            or item["status"] != "denied"
            or type(item["error"]) is not str
            for item in attempts
        ):
            raise ValueError("runtime boundary probe did not deny every direct access")
        derived_denials = tuple(
            sorted(
                (
                    RuntimeAccessDenial(
                        target_sha256=content_digest({"hostile_access_target": item["label"]}),
                        target_kind=(
                            "host_filesystem"
                            if item["label"] == "host-filesystem"
                            else "verifier_private"
                        ),
                        outcome="denied",
                        error_kind=item["error"],
                    )
                    for item in attempts
                ),
                key=lambda item: item.target_sha256,
            )
        )
        if self.access_denials != derived_denials:
            raise ValueError("runtime access denials differ from executed process response")
        if (
            self.network_outcome != response["network_status"]
            or self.mount_outcome != response["capability_status"]
        ):
            raise ValueError("runtime network or mount outcome differs from process response")
        targets = tuple(item.target_sha256 for item in self.access_denials)
        if targets != tuple(sorted(set(targets))):
            raise ValueError("runtime access denials must be sorted and unique")
        if {item.target_kind for item in self.access_denials} != {
            "verifier_private",
            "host_filesystem",
        }:
            raise ValueError("runtime access probe lacks private and host filesystem attempts")
        if not (info.context or {}).get("skip_identity") and (
            self.runtime_boundary_probe_id != self.expected_runtime_boundary_probe_id()
        ):
            raise ValueError("runtime boundary probe identity is stale")
        return self

    def expected_runtime_boundary_probe_id(self) -> str:
        return content_id(
            "v72-runtime-boundary-probe",
            self.model_dump(mode="json", exclude={"runtime_boundary_probe_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"runtime_boundary_probe_id": f"v72-runtime-boundary-probe:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            runtime_boundary_probe_id=provisional.expected_runtime_boundary_probe_id(),
            **provisional.model_dump(mode="python", exclude={"runtime_boundary_probe_id"}),
        )


class RuntimeStatusSeparationEvidence(StrictFrozenModel):
    """Distinct executed outcomes proving runtime status classification."""

    schema_version: Literal["v7.2.runtime-status-separation.v1"] = (
        "v7.2.runtime-status-separation.v1"
    )
    status_evidence_id: ContentIdentifier
    execution_profile_id: ContentIdentifier
    runtime_boundary_id: ContentIdentifier
    success_execution: JsonProcessExecution
    policy_failure_execution: JsonProcessExecution
    resource_failure_execution: JsonProcessExecution
    timeout_execution: JsonProcessExecution
    infrastructure_failure_execution: JsonProcessExecution

    @model_validator(mode="after")
    def validate_statuses(self, info: ValidationInfo) -> Self:
        executions = (
            self.success_execution,
            self.policy_failure_execution,
            self.resource_failure_execution,
            self.timeout_execution,
            self.infrastructure_failure_execution,
        )
        identities = tuple(item.execution_id for item in executions)
        if len(set(identities)) != 5:
            raise ValueError("runtime status evidence must use distinct executions")
        if tuple(item.status for item in executions) != (
            "success",
            "policy_failure",
            "resource_failure",
            "timeout",
            "infrastructure_failure",
        ):
            raise ValueError("runtime status evidence does not prove all distinct outcomes")
        if any(
            item.request.authority_context
            != {"role": "status_probe", "expected_status": item.status}
            or item.request.isolation != "policy"
            for item in executions
        ):
            raise ValueError("runtime status process authority is incomplete")
        if not (info.context or {}).get("skip_identity") and self.status_evidence_id != (
            self.expected_status_evidence_id()
        ):
            raise ValueError("runtime status separation identity is stale")
        return self

    def expected_status_evidence_id(self) -> str:
        return content_id(
            "v72-runtime-status-separation",
            self.model_dump(mode="json", exclude={"status_evidence_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"status_evidence_id": f"v72-runtime-status-separation:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            status_evidence_id=provisional.expected_status_evidence_id(),
            **provisional.model_dump(mode="python", exclude={"status_evidence_id"}),
        )


class ProductionProfileQualification(StrictFrozenModel):
    """Complete content-bound qualification of the release execution profile."""

    schema_version: Literal["v7.2.production-profile-qualification.v1"] = (
        "v7.2.production-profile-qualification.v1"
    )
    profile_qualification_id: ContentIdentifier
    execution_profile: ExecutionProfile
    execution_profile_id: ContentIdentifier
    runtime_boundary: RuntimeBoundaryAuthority
    runtime_boundary_id: ContentIdentifier
    baseline_policy: BaselinePolicyIdentity
    baseline_policy_id: ContentIdentifier
    qualified_task_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    policy_digest_set: PolicyDigestSet
    dry_run_evidence: tuple[RuntimeDryRunEvidence, ...] = Field(min_length=1)
    boundary_probe_evidence: RuntimeBoundaryProbeEvidence
    status_separation_evidence: RuntimeStatusSeparationEvidence
    process_executions: tuple[JsonProcessExecution, ...] = Field(min_length=1)
    runtime_checks: tuple[RuntimeCheckEvidence, ...]

    @model_validator(mode="after")
    def validate_qualification(self, info: ValidationInfo) -> Self:
        if self.execution_profile_id != self.execution_profile.profile_id:
            raise ValueError("production profile qualification has a stale profile reference")
        if self.runtime_boundary_id != self.runtime_boundary.runtime_boundary_id:
            raise ValueError("production profile qualification has a stale boundary reference")
        if self.baseline_policy_id != self.baseline_policy.baseline_policy_id:
            raise ValueError("production profile qualification has a stale baseline reference")
        task_ids = tuple(evidence.task_id for evidence in self.dry_run_evidence)
        if task_ids != tuple(sorted(set(task_ids))) or self.qualified_task_ids != task_ids:
            raise ValueError("runtime dry runs must cover the exact sorted task authority")
        check_kinds = tuple(check.check_kind for check in self.runtime_checks)
        if check_kinds != REQUIRED_RUNTIME_CHECKS:
            raise ValueError("production profile requires every ordered runtime check")
        if any(
            check.execution_profile_id != self.execution_profile.profile_id
            or check.runtime_boundary_id != self.runtime_boundary.runtime_boundary_id
            for check in self.runtime_checks
        ):
            raise ValueError("runtime checks do not bind the selected profile and boundary")
        if any(
            evidence.execution_profile_id != self.execution_profile.profile_id
            or evidence.runtime_boundary_id != self.runtime_boundary.runtime_boundary_id
            for evidence in self.dry_run_evidence
        ):
            raise ValueError("runtime dry runs do not bind the selected profile and boundary")
        if (
            self.boundary_probe_evidence.execution_profile_id != self.execution_profile.profile_id
            or self.boundary_probe_evidence.runtime_boundary_id
            != self.runtime_boundary.runtime_boundary_id
        ):
            raise ValueError("runtime boundary probe binds another profile or boundary")
        if (
            self.status_separation_evidence.execution_profile_id
            != self.execution_profile.profile_id
            or self.status_separation_evidence.runtime_boundary_id
            != self.runtime_boundary.runtime_boundary_id
        ):
            raise ValueError("runtime status evidence binds another profile or boundary")
        execution_ids = tuple(item.execution_id for item in self.process_executions)
        if execution_ids != tuple(sorted(set(execution_ids))):
            raise ValueError("runtime process executions must be sorted and unique")
        executions_by_id = {item.execution_id: item for item in self.process_executions}
        dry_run_ids = {
            execution_id
            for evidence in self.dry_run_evidence
            for execution_id in (
                evidence.candidate_policy_execution_id,
                evidence.final_policy_execution_id,
                evidence.candidate_verifier_execution_id,
                evidence.final_verifier_execution_id,
            )
        }
        status_executions = (
            self.status_separation_evidence.success_execution,
            self.status_separation_evidence.policy_failure_execution,
            self.status_separation_evidence.resource_failure_execution,
            self.status_separation_evidence.timeout_execution,
            self.status_separation_evidence.infrastructure_failure_execution,
        )
        expected_execution_ids = dry_run_ids | {
            self.boundary_probe_evidence.process_execution.execution_id,
            *(item.execution_id for item in status_executions),
        }
        if set(execution_ids) != expected_execution_ids:
            raise ValueError("runtime process records do not exactly cover qualification evidence")
        if any(executions_by_id[item.execution_id] != item for item in status_executions) or (
            executions_by_id[self.boundary_probe_evidence.process_execution.execution_id]
            != self.boundary_probe_evidence.process_execution
        ):
            raise ValueError("runtime embedded evidence differs from process authority")
        if any(executions_by_id[item].status != "success" for item in dry_run_ids):
            raise ValueError("runtime dry runs must reference successful process executions")
        for dry_run in self.dry_run_evidence:
            modes = (
                (
                    "candidate",
                    dry_run.candidate_workspace_id,
                    dry_run.candidate_policy_execution_id,
                    dry_run.candidate_verifier_execution_id,
                    dry_run.candidate_output_sha256,
                    dry_run.candidate_reward_sha256,
                ),
                (
                    "final",
                    dry_run.final_workspace_id,
                    dry_run.final_policy_execution_id,
                    dry_run.final_verifier_execution_id,
                    dry_run.final_output_sha256,
                    dry_run.final_reward_sha256,
                ),
            )
            for (
                mode,
                workspace_id,
                policy_execution_id,
                verifier_execution_id,
                output_sha256,
                reward_sha256,
            ) in modes:
                policy_execution = executions_by_id[policy_execution_id]
                verifier_execution = executions_by_id[verifier_execution_id]
                policy_context = policy_execution.request.authority_context
                verifier_context = verifier_execution.request.authority_context
                if (
                    type(policy_context) is not dict
                    or set(policy_context)
                    != {
                        "role",
                        "mode",
                        "task_id",
                        "workspace_manifest",
                        "baseline_policy",
                    }
                    or policy_context["role"] != "policy"
                    or policy_context["mode"] != mode
                    or policy_context["task_id"] != dry_run.task_id
                    or type(verifier_context) is not dict
                    or set(verifier_context)
                    != {
                        "role",
                        "mode",
                        "task_id",
                        "workspace_manifest",
                        "sealed_output",
                        "verifier_bundle",
                        "reward_contract",
                    }
                    or verifier_context["role"] != "verifier"
                    or verifier_context["mode"] != mode
                    or verifier_context["task_id"] != dry_run.task_id
                ):
                    raise ValueError("runtime dry-run process authority context is incomplete")
                try:
                    policy_workspace = PublicWorkspaceManifest.model_validate_json(
                        canonical_json(policy_context["workspace_manifest"])
                    )
                    observed_baseline = BaselinePolicyIdentity.model_validate_json(
                        canonical_json(policy_context["baseline_policy"])
                    )
                    verifier_workspace = PublicWorkspaceManifest.model_validate_json(
                        canonical_json(verifier_context["workspace_manifest"])
                    )
                    sealed_output = SealedOutputManifest.model_validate_json(
                        canonical_json(verifier_context["sealed_output"])
                    )
                    verifier_bundle = VerifierBundleAuthority.model_validate_json(
                        canonical_json(verifier_context["verifier_bundle"])
                    )
                    reward_contract = RewardContract.model_validate_json(
                        canonical_json(verifier_context["reward_contract"])
                    )
                    reward_result = RewardResult.model_validate_json(
                        canonical_json(verifier_execution.response)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError("runtime dry-run embedded authority is invalid") from error
                if observed_baseline != self.baseline_policy:
                    raise ValueError("runtime dry-run binds a different baseline policy")
                policy_response = policy_execution.response
                if (
                    policy_workspace != verifier_workspace
                    or policy_workspace.workspace_id != workspace_id
                    or sealed_output.public_workspace_id != workspace_id
                    or sealed_output.task_id != dry_run.task_id
                    or type(policy_response) is not dict
                    or policy_response.get("output_tree_sha256") != output_sha256
                    or policy_response.get("output_file_count") != len(sealed_output.files)
                    or sealed_output.output_tree_sha256 != output_sha256
                    or reward_result.task_id != dry_run.task_id
                    or reward_result.reward_contract_id != reward_contract.contract_id
                    or verifier_bundle.task_id != dry_run.task_id
                    or verifier_bundle.reward_contract_id != reward_contract.contract_id
                    or policy_execution.request.entrypoint != self.baseline_policy.policy_entrypoint
                    or policy_execution.request.code_closure_sha256
                    != self.baseline_policy.policy_code_closure_sha256
                    or verifier_execution.request.entrypoint != verifier_bundle.verifier_entrypoint
                    or verifier_execution.request.code_closure_sha256
                    != verifier_bundle.verifier_code_closure_sha256
                    or content_digest(reward_result.model_dump(mode="json")) != reward_sha256
                    or policy_execution.request.isolation != "policy"
                    or verifier_execution.request.isolation != "verifier"
                ):
                    raise ValueError(
                        "runtime dry-run digests differ from executed process authority"
                    )
        limits = self.execution_profile.resource_limits
        if any(
            item.elapsed_seconds > limits.maximum_elapsed_seconds
            or item.cpu_seconds > limits.maximum_cpu_seconds
            or item.peak_memory_bytes > limits.maximum_peak_memory_bytes
            or item.output_bytes > limits.maximum_disk_bytes
            or item.output_entries > limits.maximum_output_entries
            for item in self.process_executions
        ):
            raise ValueError("runtime execution measurement exceeds the production profile")
        checks_by_kind = {item.check_kind: item for item in self.runtime_checks}
        status_ids = tuple(sorted(item.execution_id for item in status_executions))
        if checks_by_kind["status_separation"].execution_evidence_ids != status_ids:
            raise ValueError("runtime status check differs from executed status evidence")
        if checks_by_kind["measured_resource_cost"].execution_evidence_ids != execution_ids:
            raise ValueError("runtime resource check differs from measured executions")
        policy_ids = tuple(
            sorted(
                execution_id
                for evidence in self.dry_run_evidence
                for execution_id in (
                    evidence.candidate_policy_execution_id,
                    evidence.final_policy_execution_id,
                )
            )
        )
        verifier_ids = tuple(
            sorted(
                execution_id
                for evidence in self.dry_run_evidence
                for execution_id in (
                    evidence.candidate_verifier_execution_id,
                    evidence.final_verifier_execution_id,
                )
            )
        )
        boundary_id = self.boundary_probe_evidence.process_execution.execution_id
        expected_check_ids = {
            "workspace_determinism": tuple(sorted(dry_run_ids)),
            "policy_private_isolation": (boundary_id,),
            "output_sealing": policy_ids,
            "verifier_private_isolation": tuple(sorted((*verifier_ids, boundary_id))),
            "status_separation": status_ids,
            "measured_resource_cost": execution_ids,
        }
        if any(
            checks_by_kind[kind].execution_evidence_ids != ids
            for kind, ids in expected_check_ids.items()
        ):
            raise ValueError("runtime checks do not bind their exact executed evidence")
        expected_check_digests = {
            "workspace_determinism": content_digest(
                [item.model_dump(mode="json") for item in self.dry_run_evidence]
            ),
            "policy_private_isolation": content_digest(
                self.boundary_probe_evidence.model_dump(mode="json")
            ),
            "output_sealing": content_digest(
                {
                    "dry_runs": [item.model_dump(mode="json") for item in self.dry_run_evidence],
                    "policy_executions": [
                        executions_by_id[item].model_dump(mode="json") for item in policy_ids
                    ],
                }
            ),
            "verifier_private_isolation": content_digest(
                {
                    "boundary_probe": self.boundary_probe_evidence.model_dump(mode="json"),
                    "verifier_executions": [
                        executions_by_id[item].model_dump(mode="json") for item in verifier_ids
                    ],
                }
            ),
            "status_separation": content_digest(
                self.status_separation_evidence.model_dump(mode="json")
            ),
            "measured_resource_cost": content_digest(
                {
                    "resource_limits": limits.model_dump(mode="json"),
                    "process_executions": [
                        item.model_dump(mode="json") for item in self.process_executions
                    ],
                }
            ),
        }
        if any(
            checks_by_kind[kind].evidence_sha256 != digest
            for kind, digest in expected_check_digests.items()
        ):
            raise ValueError("runtime check digest does not match executed evidence")
        if len({check.runtime_check_id for check in self.runtime_checks}) != len(
            self.runtime_checks
        ):
            raise ValueError("runtime check identities must be unique")
        if not (info.context or {}).get("skip_identity") and (
            self.profile_qualification_id != self.expected_profile_qualification_id()
        ):
            raise ValueError("profile qualification identity is stale")
        return self

    def expected_profile_qualification_id(self) -> str:
        return content_id(
            "v72-production-profile-qualification",
            self.model_dump(mode="json", exclude={"profile_qualification_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        execution_profile = values.get("execution_profile")
        runtime_boundary = values.get("runtime_boundary")
        baseline_policy = values.get("baseline_policy")
        dry_runs = values.get("dry_run_evidence")
        if not isinstance(execution_profile, ExecutionProfile) or not isinstance(
            runtime_boundary, RuntimeBoundaryAuthority
        ):
            raise TypeError("production qualification requires exact profile and boundary models")
        if (
            not isinstance(baseline_policy, BaselinePolicyIdentity)
            or not isinstance(dry_runs, tuple)
            or not all(isinstance(item, RuntimeDryRunEvidence) for item in dry_runs)
        ):
            raise TypeError("production qualification requires baseline and dry-run authority")
        values = {
            **values,
            "execution_profile_id": execution_profile.profile_id,
            "runtime_boundary_id": runtime_boundary.runtime_boundary_id,
            "baseline_policy_id": baseline_policy.baseline_policy_id,
            "qualified_task_ids": tuple(item.task_id for item in dry_runs),
        }
        provisional = cls.model_validate(
            {
                "profile_qualification_id": (f"v72-production-profile-qualification:{'0' * 64}"),
                **values,
            },
            context={"skip_identity": True},
        )
        return cls(
            profile_qualification_id=provisional.expected_profile_qualification_id(),
            **provisional.model_dump(mode="python", exclude={"profile_qualification_id"}),
        )


class SealedOutputFile(StrictFrozenModel):
    """One regular file in a sealed policy output tree."""

    path: NonEmptyString
    sha256: Sha256Digest
    size_bytes: NonNegativeInteger

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        safe_relative_path(self.path)
        return self


class SealedOutputManifest(StrictFrozenModel):
    """Immutable policy output authority established before verifier startup."""

    schema_version: Literal["v7.2.sealed-output.v1"] = "v7.2.sealed-output.v1"
    sealed_output_id: ContentIdentifier
    task_id: ContentIdentifier
    public_workspace_id: ContentIdentifier
    output_tree_sha256: Sha256Digest
    files: tuple[SealedOutputFile, ...]

    @model_validator(mode="after")
    def validate_output(self, info: ValidationInfo) -> Self:
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("sealed output paths must be sorted and unique")
        authority = [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in self.files
        ]
        if self.output_tree_sha256 != content_digest(authority):
            raise ValueError("sealed output tree digest does not match file authority")
        if not (info.context or {}).get("skip_identity") and (
            self.sealed_output_id != self.expected_sealed_output_id()
        ):
            raise ValueError("sealed_output_id does not match output authority")
        return self

    def expected_sealed_output_id(self) -> str:
        return content_id(
            "v72-sealed-output",
            self.model_dump(mode="json", exclude={"sealed_output_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"sealed_output_id": f"v72-sealed-output:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            sealed_output_id=provisional.expected_sealed_output_id(),
            **provisional.model_dump(mode="python", exclude={"sealed_output_id"}),
        )


class VerifierInvocationAuthority(StrictFrozenModel):
    """Exact public, sealed-output, and private inputs allowed at verifier startup."""

    schema_version: Literal["v7.2.verifier-invocation.v1"] = "v7.2.verifier-invocation.v1"
    verifier_invocation_id: ContentIdentifier
    candidate_release_id: ContentIdentifier
    rollout_id: ContentIdentifier
    task_id: ContentIdentifier
    execution_profile_id: ContentIdentifier
    runtime_boundary_id: ContentIdentifier
    public_workspace_id: ContentIdentifier
    sealed_output_id: ContentIdentifier
    verifier_bundle_id: ContentIdentifier
    verifier_private_tree_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_invocation(self, info: ValidationInfo) -> Self:
        if not (info.context or {}).get("skip_identity") and (
            self.verifier_invocation_id != self.expected_verifier_invocation_id()
        ):
            raise ValueError("verifier invocation identity is stale")
        return self

    def expected_verifier_invocation_id(self) -> str:
        return content_id(
            "v72-verifier-invocation",
            self.model_dump(mode="json", exclude={"verifier_invocation_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"verifier_invocation_id": f"v72-verifier-invocation:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            verifier_invocation_id=provisional.expected_verifier_invocation_id(),
            **provisional.model_dump(mode="python", exclude={"verifier_invocation_id"}),
        )


__all__ = [
    "PUBLIC_RUNTIME_PATHS",
    "REQUIRED_RUNTIME_CHECKS",
    "REQUIRED_RUNTIME_ENVIRONMENT_VARIABLES",
    "BaselinePolicyIdentity",
    "ExecutionProfile",
    "ExecutionResourceLimits",
    "ProductionProfileQualification",
    "RuntimeAccessDenial",
    "RuntimeBoundaryAuthority",
    "RuntimeBoundaryProbeEvidence",
    "RuntimeCheckEvidence",
    "RuntimeDryRunEvidence",
    "RuntimeStatusSeparationEvidence",
    "SealedOutputFile",
    "SealedOutputManifest",
    "VerifierBundleAuthority",
    "VerifierInvocationAuthority",
]
