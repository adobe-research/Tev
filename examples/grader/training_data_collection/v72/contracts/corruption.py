"""Controlled-corruption authority contracts for v7.2 generation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import content_id

from .identity import ContentIdentifier, Sha256Digest, StrictFrozenModel

CORRUPTION_OPERATOR_SCHEMA_VERSION = "v7.2.corruption-operator.v1"
CORRUPTION_EVENT_SCHEMA_VERSION = "v7.2.corruption-event.v1"


def _require_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{label} must contain nonblank values")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


class _CorruptionOperatorAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.corruption-operator.v1"] = CORRUPTION_OPERATOR_SCHEMA_VERSION
    operator_id: str = Field(min_length=1)
    implementation_digest: Sha256Digest
    affected_artifact_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    affected_relation_ids: tuple[ContentIdentifier, ...] = ()
    affected_invariant_ids: tuple[ContentIdentifier, ...] = ()
    affected_stage_ids: tuple[str, ...] = ()
    expected_metamorphic_effects: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operator_authority(self) -> Self:
        if not self.operator_id.strip():
            raise ValueError("corruption operator identity cannot be blank")
        for label, values in (
            ("affected artifact identities", self.affected_artifact_ids),
            ("affected relation identities", self.affected_relation_ids),
            ("affected invariant identities", self.affected_invariant_ids),
            ("affected stage identities", self.affected_stage_ids),
            ("expected metamorphic effects", self.expected_metamorphic_effects),
        ):
            if values:
                _require_sorted_unique(values, label)
        if not self.affected_relation_ids and not self.affected_invariant_ids:
            raise ValueError("corruption must affect a registered relation or invariant")
        return self


class CorruptionOperatorDeclaration(_CorruptionOperatorAuthority):
    """Content-bound registration for one deterministic corruption implementation."""

    declaration_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_declaration_id(self) -> Self:
        if self.declaration_id != self.expected_declaration_id():
            raise ValueError("declaration_id does not match corruption operator authority")
        return self

    def expected_declaration_id(self) -> str:
        return content_id(
            "v72-corruption-operator",
            self.model_dump(mode="json", exclude={"declaration_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> CorruptionOperatorDeclaration:
        authority = _CorruptionOperatorAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            declaration_id=content_id("v72-corruption-operator", authority.model_dump(mode="json")),
        )


class _CorruptionEventAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.corruption-event.v1"] = CORRUPTION_EVENT_SCHEMA_VERSION
    world_id: ContentIdentifier
    operator_id: str = Field(min_length=1)
    operator_declaration_id: ContentIdentifier
    baseline_artifact_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    affected_artifact_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    affected_relation_ids: tuple[ContentIdentifier, ...] = ()
    affected_invariant_ids: tuple[ContentIdentifier, ...] = ()
    affected_stage_ids: tuple[str, ...] = ()
    expected_metamorphic_effects: tuple[str, ...] = Field(min_length=1)
    seed_namespace: Literal["perturbation"] = "perturbation"
    baseline_tree_sha256: Sha256Digest
    corrupted_tree_sha256: Sha256Digest
    changed_paths: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_event_authority(self) -> Self:
        if not self.operator_id.strip():
            raise ValueError("corruption event operator identity cannot be blank")
        for label, values in (
            ("baseline artifact identities", self.baseline_artifact_ids),
            ("affected artifact identities", self.affected_artifact_ids),
            ("affected relation identities", self.affected_relation_ids),
            ("affected invariant identities", self.affected_invariant_ids),
            ("affected stage identities", self.affected_stage_ids),
            ("expected metamorphic effects", self.expected_metamorphic_effects),
            ("changed paths", self.changed_paths),
        ):
            if values:
                _require_sorted_unique(values, label)
        if not set(self.affected_artifact_ids).issubset(self.baseline_artifact_ids):
            raise ValueError("affected artifacts must belong to the baseline artifact set")
        for path in self.changed_paths:
            safe_relative_path(path)
        if self.baseline_tree_sha256 == self.corrupted_tree_sha256:
            raise ValueError("registered corruption must change the realized artifact tree")
        return self


class CorruptionEvent(_CorruptionEventAuthority):
    """Realized, content-bound effects of one registered corruption."""

    corruption_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_corruption_id(self) -> Self:
        if self.corruption_id != self.expected_corruption_id():
            raise ValueError("corruption_id does not match realized corruption authority")
        return self

    def expected_corruption_id(self) -> str:
        return content_id(
            "v72-corruption-event",
            self.model_dump(mode="json", exclude={"corruption_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> CorruptionEvent:
        authority = _CorruptionEventAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            corruption_id=content_id("v72-corruption-event", authority.model_dump(mode="json")),
        )


__all__ = [
    "CORRUPTION_EVENT_SCHEMA_VERSION",
    "CORRUPTION_OPERATOR_SCHEMA_VERSION",
    "CorruptionEvent",
    "CorruptionOperatorDeclaration",
]
