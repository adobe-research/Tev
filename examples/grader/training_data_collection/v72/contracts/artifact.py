"""Content-identified artifact graph contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import JsonScalar, canonical_json, content_id
from training_data_collection.v72.contracts.identity import ContentIdentifier, StrictFrozenModel

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInteger = Annotated[int, Field(ge=0)]

type ArtifactVisibility = Literal["public", "verifier_private"]
type ArtifactPathKind = Literal["file", "directory", "glob"]
type ArtifactSemanticRole = Literal[
    "authoritative_source",
    "raw_observation_or_log",
    "lossy_projection",
    "stale_derived_state",
    "controlled_corrupted_artifact",
    "configuration",
    "executable_or_harness",
    "schema_or_interface_contract",
    "protected_state",
    "distractor",
    "intermediate_state",
    "output_template",
    "required_output",
    "evidence_artifact",
]
type ArtifactRelationType = Literal[
    "derived_from",
    "generated_from",
    "summarizes",
    "indexes",
    "shards",
    "reassembles",
    "joins_on",
    "aligns_with",
    "must_match",
    "hashes",
    "constrained_by",
    "contradicts",
    "supersedes",
    "must_preserve",
    "protected_by",
    "validated_by",
    "consumed_by",
    "produced_by",
]


class RelationParameter(StrictFrozenModel):
    name: NonEmptyString
    value: JsonScalar


class ArtifactNode(StrictFrozenModel):
    schema_version: Literal["v7.2.artifact-node.v1"]
    artifact_id: ContentIdentifier
    visibility: ArtifactVisibility
    path: NonEmptyString
    path_kind: ArtifactPathKind
    artifact_type: NonEmptyString
    native_schema: NonEmptyString
    semantic_role: ArtifactSemanticRole
    world_projection_id: ContentIdentifier
    authority_rank: NonNegativeInteger
    mutable_by_policy: bool
    source_record_ids: tuple[ContentIdentifier, ...]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        safe_relative_path(value)
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"artifact_id"})
        if self.artifact_id != content_id("artifact-node", authority):
            raise ValueError("artifact_id does not match artifact node authority")
        if self.visibility == "verifier_private" and self.mutable_by_policy:
            raise ValueError("verifier-private artifacts cannot be policy mutable")
        return self

    @classmethod
    def create(
        cls,
        *,
        visibility: ArtifactVisibility,
        path: str,
        path_kind: ArtifactPathKind,
        artifact_type: str,
        native_schema: str,
        semantic_role: ArtifactSemanticRole,
        world_projection_id: str,
        authority_rank: int,
        mutable_by_policy: bool,
        source_record_ids: tuple[str, ...],
    ) -> Self:
        authority = {
            "schema_version": "v7.2.artifact-node.v1",
            "visibility": visibility,
            "path": path,
            "path_kind": path_kind,
            "artifact_type": artifact_type,
            "native_schema": native_schema,
            "semantic_role": semantic_role,
            "world_projection_id": world_projection_id,
            "authority_rank": authority_rank,
            "mutable_by_policy": mutable_by_policy,
            "source_record_ids": list(source_record_ids),
        }
        return cls.model_validate_json(
            canonical_json({"artifact_id": content_id("artifact-node", authority), **authority})
        )


class ArtifactRelation(StrictFrozenModel):
    schema_version: Literal["v7.2.artifact-relation.v1"]
    relation_id: ContentIdentifier
    relation_type: ArtifactRelationType
    source_artifact_ids: tuple[ContentIdentifier, ...]
    target_artifact_ids: tuple[ContentIdentifier, ...]
    key_parameters: tuple[RelationParameter, ...]
    public_rule_ids: tuple[NonEmptyString, ...]
    invariant_ids: tuple[NonEmptyString, ...]
    workflow_stage_ids: tuple[NonEmptyString, ...]
    reward_component_ids: tuple[NonEmptyString, ...]
    validator_implementation_id: NonEmptyString
    critical: bool

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.source_artifact_ids or not self.target_artifact_ids:
            raise ValueError("artifact relations require source and target artifacts")
        if len(set(self.source_artifact_ids)) != len(self.source_artifact_ids):
            raise ValueError("artifact relation source endpoints must be unique")
        if len(set(self.target_artifact_ids)) != len(self.target_artifact_ids):
            raise ValueError("artifact relation target endpoints must be unique")
        authority = self.model_dump(mode="json", exclude={"relation_id"})
        if self.relation_id != content_id("artifact-relation", authority):
            raise ValueError("relation_id does not match artifact relation authority")
        return self

    @classmethod
    def create(
        cls,
        *,
        relation_type: ArtifactRelationType,
        source_artifact_ids: tuple[str, ...],
        target_artifact_ids: tuple[str, ...],
        key_parameters: tuple[RelationParameter, ...],
        public_rule_ids: tuple[str, ...],
        invariant_ids: tuple[str, ...],
        workflow_stage_ids: tuple[str, ...],
        reward_component_ids: tuple[str, ...],
        validator_implementation_id: str,
        critical: bool,
    ) -> Self:
        authority = {
            "schema_version": "v7.2.artifact-relation.v1",
            "relation_type": relation_type,
            "source_artifact_ids": list(source_artifact_ids),
            "target_artifact_ids": list(target_artifact_ids),
            "key_parameters": [item.model_dump(mode="json") for item in key_parameters],
            "public_rule_ids": list(public_rule_ids),
            "invariant_ids": list(invariant_ids),
            "workflow_stage_ids": list(workflow_stage_ids),
            "reward_component_ids": list(reward_component_ids),
            "validator_implementation_id": validator_implementation_id,
            "critical": critical,
        }
        return cls.model_validate_json(
            canonical_json({"relation_id": content_id("artifact-relation", authority), **authority})
        )


class ApprovedDerivationCycle(StrictFrozenModel):
    schema_version: Literal["v7.2.approved-derivation-cycle.v1"]
    cycle_id: ContentIdentifier
    relation_ids: tuple[ContentIdentifier, ...]
    approval_record_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_relation_set(self) -> Self:
        if not self.relation_ids or len(set(self.relation_ids)) != len(self.relation_ids):
            raise ValueError("approved derivation cycle relation IDs must be nonempty and unique")
        authority = self.model_dump(mode="json", exclude={"cycle_id"})
        if self.cycle_id != content_id("derivation-cycle", authority):
            raise ValueError("cycle_id does not match derivation cycle authority")
        return self

    @classmethod
    def create(cls, *, relation_ids: tuple[str, ...], approval_record_id: str) -> Self:
        authority = {
            "schema_version": "v7.2.approved-derivation-cycle.v1",
            "relation_ids": list(relation_ids),
            "approval_record_id": approval_record_id,
        }
        return cls.model_validate_json(
            canonical_json({"cycle_id": content_id("derivation-cycle", authority), **authority})
        )


class ArtifactGraph(StrictFrozenModel):
    schema_version: Literal["v7.2.artifact-graph.v1"]
    graph_id: ContentIdentifier
    world_id: ContentIdentifier
    nodes: tuple[ArtifactNode, ...]
    relations: tuple[ArtifactRelation, ...]
    critical_public_artifact_ids: tuple[ContentIdentifier, ...]
    terminal_output_artifact_ids: tuple[ContentIdentifier, ...]
    approved_derivation_cycles: tuple[ApprovedDerivationCycle, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"graph_id"})
        if self.graph_id != content_id("artifact-graph", authority):
            raise ValueError("graph_id does not match artifact graph authority")
        return self

    @classmethod
    def create(
        cls,
        *,
        world_id: str,
        nodes: tuple[ArtifactNode, ...],
        relations: tuple[ArtifactRelation, ...],
        critical_public_artifact_ids: tuple[str, ...],
        terminal_output_artifact_ids: tuple[str, ...],
        approved_derivation_cycles: tuple[ApprovedDerivationCycle, ...] = (),
    ) -> Self:
        authority = {
            "schema_version": "v7.2.artifact-graph.v1",
            "world_id": world_id,
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "relations": [relation.model_dump(mode="json") for relation in relations],
            "critical_public_artifact_ids": list(critical_public_artifact_ids),
            "terminal_output_artifact_ids": list(terminal_output_artifact_ids),
            "approved_derivation_cycles": [
                cycle.model_dump(mode="json") for cycle in approved_derivation_cycles
            ],
        }
        return cls.model_validate_json(
            canonical_json({"graph_id": content_id("artifact-graph", authority), **authority})
        )
