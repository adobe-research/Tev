"""Content-identified workflow DAG contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from training_data_collection.common.identity import canonical_json, content_id
from training_data_collection.v72.contracts.identity import ContentIdentifier, StrictFrozenModel

NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]

type CurriculumLayer = Literal["primitive_terminal", "family_integration", "professional_frontier"]
type EvidenceObservationKind = Literal[
    "recomputed_artifact", "runtime_observation", "state_transition", "self_report"
]


class FailureTransition(StrictFrozenModel):
    schema_version: Literal["v7.2.failure-transition.v1"]
    transition_id: ContentIdentifier
    from_state_id: NonEmptyString
    to_state_id: NonEmptyString
    trigger: NonEmptyString

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"transition_id"})
        if self.transition_id != content_id("failure-transition", authority):
            raise ValueError("transition_id does not match failure transition authority")
        if self.from_state_id == self.to_state_id:
            raise ValueError("failure transition must change observable state")
        return self

    @classmethod
    def create(cls, *, from_state_id: str, to_state_id: str, trigger: str) -> Self:
        authority = {
            "schema_version": "v7.2.failure-transition.v1",
            "from_state_id": from_state_id,
            "to_state_id": to_state_id,
            "trigger": trigger,
        }
        return cls.model_validate_json(
            canonical_json(
                {"transition_id": content_id("failure-transition", authority), **authority}
            )
        )


class VerifierEvidenceDeclaration(StrictFrozenModel):
    schema_version: Literal["v7.2.verifier-evidence-declaration.v1"]
    evidence_id: ContentIdentifier
    observation_kind: EvidenceObservationKind
    implementation_id: NonEmptyString
    observed_artifact_ids: tuple[ContentIdentifier, ...]
    observed_state_ids: tuple[NonEmptyString, ...]
    independently_recomputed: bool

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.observed_artifact_ids and not self.observed_state_ids:
            raise ValueError("verifier evidence must observe an artifact or state")
        authority = self.model_dump(mode="json", exclude={"evidence_id"})
        if self.evidence_id != content_id("verifier-evidence", authority):
            raise ValueError("evidence_id does not match verifier evidence authority")
        return self

    @classmethod
    def create(
        cls,
        *,
        observation_kind: EvidenceObservationKind,
        implementation_id: str,
        observed_artifact_ids: tuple[str, ...],
        observed_state_ids: tuple[str, ...],
        independently_recomputed: bool,
    ) -> Self:
        authority = {
            "schema_version": "v7.2.verifier-evidence-declaration.v1",
            "observation_kind": observation_kind,
            "implementation_id": implementation_id,
            "observed_artifact_ids": list(observed_artifact_ids),
            "observed_state_ids": list(observed_state_ids),
            "independently_recomputed": independently_recomputed,
        }
        return cls.model_validate_json(
            canonical_json({"evidence_id": content_id("verifier-evidence", authority), **authority})
        )


class WorkflowStage(StrictFrozenModel):
    schema_version: Literal["v7.2.workflow-stage.v1"]
    stage_record_id: ContentIdentifier
    stage_id: NonEmptyString
    description: NonEmptyString
    depends_on_stage_ids: tuple[NonEmptyString, ...]
    consumed_artifact_ids: tuple[ContentIdentifier, ...]
    consumed_relation_ids: tuple[ContentIdentifier, ...]
    required_tool_capabilities: tuple[NonEmptyString, ...]
    input_state_ids: tuple[NonEmptyString, ...]
    produced_state_ids: tuple[NonEmptyString, ...]
    produced_artifact_ids: tuple[ContentIdentifier, ...]
    expected_failure_transitions: tuple[FailureTransition, ...]
    verifier_evidence_ids: tuple[ContentIdentifier, ...]
    reward_component_ids: tuple[NonEmptyString, ...]
    minimum_semantic_work: PositiveInteger
    ablation_implementation_id: NonEmptyString

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.stage_id in self.depends_on_stage_ids:
            raise ValueError("workflow stage cannot depend on itself")
        for values in (
            self.depends_on_stage_ids,
            self.consumed_artifact_ids,
            self.consumed_relation_ids,
            self.input_state_ids,
            self.produced_state_ids,
            self.produced_artifact_ids,
            self.verifier_evidence_ids,
            self.reward_component_ids,
        ):
            if len(set(values)) != len(values):
                raise ValueError("workflow stage reference sets must be unique")
        if not self.produced_state_ids and not self.produced_artifact_ids:
            raise ValueError("workflow stage must produce independently observable state")
        authority = self.model_dump(mode="json", exclude={"stage_record_id"})
        if self.stage_record_id != content_id("workflow-stage", authority):
            raise ValueError("stage_record_id does not match workflow stage authority")
        return self

    @classmethod
    def create(
        cls,
        *,
        stage_id: str,
        description: str,
        depends_on_stage_ids: tuple[str, ...],
        consumed_artifact_ids: tuple[str, ...],
        consumed_relation_ids: tuple[str, ...],
        required_tool_capabilities: tuple[str, ...],
        input_state_ids: tuple[str, ...],
        produced_state_ids: tuple[str, ...],
        produced_artifact_ids: tuple[str, ...],
        expected_failure_transitions: tuple[FailureTransition, ...],
        verifier_evidence_ids: tuple[str, ...],
        reward_component_ids: tuple[str, ...],
        minimum_semantic_work: int,
        ablation_implementation_id: str,
    ) -> Self:
        authority = {
            "schema_version": "v7.2.workflow-stage.v1",
            "stage_id": stage_id,
            "description": description,
            "depends_on_stage_ids": list(depends_on_stage_ids),
            "consumed_artifact_ids": list(consumed_artifact_ids),
            "consumed_relation_ids": list(consumed_relation_ids),
            "required_tool_capabilities": list(required_tool_capabilities),
            "input_state_ids": list(input_state_ids),
            "produced_state_ids": list(produced_state_ids),
            "produced_artifact_ids": list(produced_artifact_ids),
            "expected_failure_transitions": [
                transition.model_dump(mode="json") for transition in expected_failure_transitions
            ],
            "verifier_evidence_ids": list(verifier_evidence_ids),
            "reward_component_ids": list(reward_component_ids),
            "minimum_semantic_work": minimum_semantic_work,
            "ablation_implementation_id": ablation_implementation_id,
        }
        return cls.model_validate_json(
            canonical_json(
                {"stage_record_id": content_id("workflow-stage", authority), **authority}
            )
        )


class AllowedParallelGroup(StrictFrozenModel):
    stage_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if len(self.stage_ids) < 2 or len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValueError("parallel groups require at least two unique stage IDs")
        return self


class WorkflowGraph(StrictFrozenModel):
    schema_version: Literal["v7.2.workflow-graph.v1"]
    workflow_id: ContentIdentifier
    artifact_graph_id: ContentIdentifier
    curriculum_layer: CurriculumLayer
    initial_state_ids: tuple[NonEmptyString, ...]
    stages: tuple[WorkflowStage, ...]
    verifier_evidence: tuple[VerifierEvidenceDeclaration, ...]
    terminal_output_artifact_ids: tuple[ContentIdentifier, ...]
    critical_stage_ids: tuple[NonEmptyString, ...]
    allowed_parallel_groups: tuple[AllowedParallelGroup, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"workflow_id"})
        if self.workflow_id != content_id("workflow-graph", authority):
            raise ValueError("workflow_id does not match workflow graph authority")
        return self

    @classmethod
    def create(
        cls,
        *,
        artifact_graph_id: str,
        curriculum_layer: CurriculumLayer,
        initial_state_ids: tuple[str, ...],
        stages: tuple[WorkflowStage, ...],
        verifier_evidence: tuple[VerifierEvidenceDeclaration, ...],
        terminal_output_artifact_ids: tuple[str, ...],
        critical_stage_ids: tuple[str, ...],
        allowed_parallel_groups: tuple[AllowedParallelGroup, ...] = (),
    ) -> Self:
        authority = {
            "schema_version": "v7.2.workflow-graph.v1",
            "artifact_graph_id": artifact_graph_id,
            "curriculum_layer": curriculum_layer,
            "initial_state_ids": list(initial_state_ids),
            "stages": [stage.model_dump(mode="json") for stage in stages],
            "verifier_evidence": [item.model_dump(mode="json") for item in verifier_evidence],
            "terminal_output_artifact_ids": list(terminal_output_artifact_ids),
            "critical_stage_ids": list(critical_stage_ids),
            "allowed_parallel_groups": [
                group.model_dump(mode="json") for group in allowed_parallel_groups
            ],
        }
        return cls.model_validate_json(
            canonical_json({"workflow_id": content_id("workflow-graph", authority), **authority})
        )
