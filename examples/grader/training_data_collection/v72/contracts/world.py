"""Latent-world authority contracts for v7.2 task generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, JsonValue, ValidationInfo, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import canonical_json, content_id, derive_seed

from .identity import ContentIdentifier, SeedNamespace, Sha256Digest, StrictFrozenModel
from .implementation import ImplementationAuthority

WORLD_RECIPE_SCHEMA_VERSION = "v7.2.world-recipe.v1"
WORLD_INVARIANT_SCHEMA_VERSION = "v7.2.world-invariant.v1"
LATENT_WORLD_SCHEMA_VERSION = "v7.2.latent-world.v1"
FAMILY_REVISION_SCHEMA_VERSION = "v7.2.family-revision.v1"


def _require_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{label} must contain nonblank values")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


class FamilyRevisionAuthority(StrictFrozenModel):
    """Canonical logical revision and exact four-role implementation authority."""

    schema_version: Literal["v7.2.family-revision.v1"] = FAMILY_REVISION_SCHEMA_VERSION
    family_revision_id: ContentIdentifier
    logical_revision: str = Field(min_length=1)
    family_implementation_entrypoint: str = Field(min_length=1)
    family_implementation_sha256: Sha256Digest
    declared_implementation_sha256: Sha256Digest
    materializer_authority: ImplementationAuthority
    reference_authority: ImplementationAuthority
    alternative_authority: ImplementationAuthority
    verifier_authority: ImplementationAuthority

    @model_validator(mode="after")
    def validate_identity(self, info: ValidationInfo) -> Self:
        if not self.logical_revision.strip():
            raise ValueError("family logical revision cannot be blank")
        authorities = (
            self.materializer_authority,
            self.reference_authority,
            self.alternative_authority,
            self.verifier_authority,
        )
        if tuple(item.authority_kind for item in authorities) != (
            "materializer",
            "reference",
            "alternative",
            "verifier",
        ):
            raise ValueError("family revision implementation roles are incomplete")
        if len({item.authority_id for item in authorities}) != len(authorities):
            raise ValueError("family revision implementation authorities must be distinct")
        if not (info.context or {}).get("skip_identity") and (
            self.family_revision_id != self.expected_family_revision_id()
        ):
            raise ValueError("family revision identity is stale")
        return self

    def expected_family_revision_id(self) -> str:
        return content_id(
            "v72-family-revision",
            self.model_dump(mode="json", exclude={"family_revision_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"family_revision_id": f"v72-family-revision:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            family_revision_id=provisional.expected_family_revision_id(),
            **provisional.model_dump(mode="python", exclude={"family_revision_id"}),
        )


class WorldParameter(StrictFrozenModel):
    """One immutable canonical JSON recipe parameter."""

    name: str = Field(min_length=1)
    value_json: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_parameter(self) -> Self:
        if not self.name.strip():
            raise ValueError("world parameter name cannot be blank")
        try:
            value = json.loads(self.value_json)
        except json.JSONDecodeError as error:
            raise ValueError("world parameter must contain valid JSON") from error
        if canonical_json(value).decode("utf-8") != self.value_json:
            raise ValueError("world parameter JSON must use canonical serialization")
        return self

    @classmethod
    def from_value(cls, name: str, value: JsonValue) -> WorldParameter:
        return cls(name=name, value_json=canonical_json(value).decode("utf-8"))


class PolicyDigestBinding(StrictFrozenModel):
    """Bind one complete policy model to a world recipe."""

    policy_id: str = Field(min_length=1)
    policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_policy_id(self) -> Self:
        if not self.policy_id.strip():
            raise ValueError("policy identity cannot be blank")
        return self


class _WorldRecipeAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.world-recipe.v1"] = WORLD_RECIPE_SCHEMA_VERSION
    family_revision_id: str = Field(min_length=1)
    root_seed: int = Field(ge=0, le=2**256 - 1)
    public_seed: int = Field(ge=0, lt=2**128)
    private_world_seed: int = Field(ge=0, lt=2**128)
    hidden_case_seed: int = Field(ge=0, lt=2**128)
    perturbation_seed: int = Field(ge=0, lt=2**128)
    behavior_seed: int = Field(ge=0, lt=2**128)
    scale_id: str = Field(min_length=1)
    horizon_target_id: str = Field(min_length=1)
    curriculum_layer: Literal["primitive_terminal", "family_integration", "professional_frontier"]
    source_record_ids: tuple[ContentIdentifier, ...] = ()
    parameter_values: tuple[WorldParameter, ...] = ()
    policy_digest_set: tuple[PolicyDigestBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        for label, value in (
            ("family revision", self.family_revision_id),
            ("scale", self.scale_id),
            ("horizon target", self.horizon_target_id),
        ):
            if not value.strip():
                raise ValueError(f"{label} identity cannot be blank")
        if self.source_record_ids:
            _require_sorted_unique(self.source_record_ids, "source record identities")
        parameter_names = tuple(parameter.name for parameter in self.parameter_values)
        if parameter_names != tuple(sorted(parameter_names)) or len(parameter_names) != len(
            set(parameter_names)
        ):
            raise ValueError("world parameters must be sorted by unique name")
        policy_ids = tuple(binding.policy_id for binding in self.policy_digest_set)
        if policy_ids != tuple(sorted(policy_ids)) or len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy digest bindings must be sorted by unique identity")

        expected_seeds = (
            derive_seed(self.root_seed, SeedNamespace.PUBLIC),
            derive_seed(self.root_seed, SeedNamespace.PRIVATE_WORLD),
            derive_seed(self.root_seed, SeedNamespace.HIDDEN_CASE),
            derive_seed(self.root_seed, SeedNamespace.PERTURBATION),
            derive_seed(self.root_seed, SeedNamespace.BEHAVIOR),
        )
        actual_seeds = (
            self.public_seed,
            self.private_world_seed,
            self.hidden_case_seed,
            self.perturbation_seed,
            self.behavior_seed,
        )
        if actual_seeds != expected_seeds:
            raise ValueError("world recipe seeds do not match their cryptographic namespaces")
        if len(set(actual_seeds)) != len(actual_seeds):
            raise ValueError("world recipe seed namespaces must be distinct")
        return self


class WorldRecipe(_WorldRecipeAuthority):
    """Complete deterministic authority for one private latent world."""

    recipe_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_recipe_id(self) -> Self:
        if self.recipe_id != self.expected_recipe_id():
            raise ValueError("recipe_id does not match world recipe authority")
        return self

    def expected_recipe_id(self) -> str:
        return content_id(
            "v72-world-recipe",
            self.model_dump(mode="json", exclude={"recipe_id"}),
        )

    def parameter_dict(self) -> dict[str, JsonValue]:
        return {
            parameter.name: json.loads(parameter.value_json) for parameter in self.parameter_values
        }

    @classmethod
    def identified(
        cls,
        *,
        family_revision_id: str,
        root_seed: int,
        scale_id: str,
        horizon_target_id: str,
        curriculum_layer: Literal[
            "primitive_terminal", "family_integration", "professional_frontier"
        ],
        policy_digests: Mapping[str, str],
        source_record_ids: tuple[str, ...] = (),
        parameter_values: Mapping[str, JsonValue] | None = None,
    ) -> WorldRecipe:
        parameters = tuple(
            WorldParameter.from_value(name, value)
            for name, value in sorted((parameter_values or {}).items())
        )
        policies = tuple(
            PolicyDigestBinding(policy_id=policy_id, policy_sha256=digest)
            for policy_id, digest in sorted(policy_digests.items())
        )
        authority = _WorldRecipeAuthority(
            family_revision_id=family_revision_id,
            root_seed=root_seed,
            public_seed=derive_seed(root_seed, SeedNamespace.PUBLIC),
            private_world_seed=derive_seed(root_seed, SeedNamespace.PRIVATE_WORLD),
            hidden_case_seed=derive_seed(root_seed, SeedNamespace.HIDDEN_CASE),
            perturbation_seed=derive_seed(root_seed, SeedNamespace.PERTURBATION),
            behavior_seed=derive_seed(root_seed, SeedNamespace.BEHAVIOR),
            scale_id=scale_id,
            horizon_target_id=horizon_target_id,
            curriculum_layer=curriculum_layer,
            source_record_ids=tuple(sorted(source_record_ids)),
            parameter_values=parameters,
            policy_digest_set=policies,
        )
        values = authority.model_dump(mode="python")
        return cls(
            **values,
            recipe_id=content_id("v72-world-recipe", authority.model_dump(mode="json")),
        )


class _WorldInvariantAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.world-invariant.v1"] = WORLD_INVARIANT_SCHEMA_VERSION
    domain_description: str = Field(min_length=1)
    validator_implementation_id: ContentIdentifier
    affected_entity_ids: tuple[str, ...] = ()
    affected_artifact_ids: tuple[ContentIdentifier, ...] = ()
    related_public_rule_ids: tuple[str, ...] = ()
    related_workflow_stage_ids: tuple[str, ...] = ()
    affected_reward_component_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_invariant_authority(self) -> Self:
        if not self.domain_description.strip():
            raise ValueError("world invariant description cannot be blank")
        for label, values in (
            ("affected entity identities", self.affected_entity_ids),
            ("affected artifact identities", self.affected_artifact_ids),
            ("public rule identities", self.related_public_rule_ids),
            ("workflow stage identities", self.related_workflow_stage_ids),
            ("reward component identities", self.affected_reward_component_ids),
        ):
            if values:
                _require_sorted_unique(values, label)
        if not self.affected_entity_ids and not self.affected_artifact_ids:
            raise ValueError("world invariant must affect an entity or artifact")
        return self


class WorldInvariant(_WorldInvariantAuthority):
    """A world truth that must have an executable realized-state validator."""

    invariant_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_invariant_id(self) -> Self:
        if self.invariant_id != self.expected_invariant_id():
            raise ValueError("invariant_id does not match world invariant authority")
        return self

    def expected_invariant_id(self) -> str:
        return content_id(
            "v72-world-invariant",
            self.model_dump(mode="json", exclude={"invariant_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> WorldInvariant:
        authority = _WorldInvariantAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            invariant_id=content_id("v72-world-invariant", authority.model_dump(mode="json")),
        )


class WorldEntityState(StrictFrozenModel):
    """Bind one world entity to its realized private-state path."""

    entity_id: str = Field(min_length=1)
    state_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entity(self) -> Self:
        if not self.entity_id.strip():
            raise ValueError("world entity identity cannot be blank")
        safe_relative_path(self.state_path)
        return self


class _LatentWorldAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.latent-world.v1"] = LATENT_WORLD_SCHEMA_VERSION
    recipe_id: ContentIdentifier
    family_revision_id: str = Field(min_length=1)
    entity_index: tuple[WorldEntityState, ...] = Field(min_length=1)
    invariant_declarations: tuple[WorldInvariant, ...] = Field(min_length=1)
    state_tree_sha256: Sha256Digest
    renderer_declarations: tuple[str, ...] = Field(min_length=1)
    source_record_ids: tuple[ContentIdentifier, ...] = ()
    creation_implementation_digest: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest_authority(self) -> Self:
        if not self.family_revision_id.strip():
            raise ValueError("family revision identity cannot be blank")
        entity_ids = tuple(entity.entity_id for entity in self.entity_index)
        entity_paths = tuple(entity.state_path for entity in self.entity_index)
        if entity_ids != tuple(sorted(entity_ids)) or len(entity_ids) != len(set(entity_ids)):
            raise ValueError("world entity index must be sorted by unique identity")
        if len(entity_paths) != len(set(entity_paths)):
            raise ValueError("world entity state paths must be unique")
        invariant_ids = tuple(invariant.invariant_id for invariant in self.invariant_declarations)
        if invariant_ids != tuple(sorted(invariant_ids)) or len(invariant_ids) != len(
            set(invariant_ids)
        ):
            raise ValueError("world invariants must be sorted by unique identity")
        unknown_entities = {
            entity_id
            for invariant in self.invariant_declarations
            for entity_id in invariant.affected_entity_ids
            if entity_id not in set(entity_ids)
        }
        if unknown_entities:
            raise ValueError(f"world invariants reference unknown entities: {unknown_entities}")
        _require_sorted_unique(self.renderer_declarations, "renderer declarations")
        if self.source_record_ids:
            _require_sorted_unique(self.source_record_ids, "source record identities")
        return self


class LatentWorldManifest(_LatentWorldAuthority):
    """Content identity and executable declarations for realized private state."""

    world_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_world_id(self) -> Self:
        if self.world_id != self.expected_world_id():
            raise ValueError("world_id does not match latent world authority")
        return self

    def expected_world_id(self) -> str:
        return content_id(
            "v72-latent-world",
            self.model_dump(mode="json", exclude={"world_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> LatentWorldManifest:
        authority = _LatentWorldAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            world_id=content_id("v72-latent-world", authority.model_dump(mode="json")),
        )


__all__ = [
    "FAMILY_REVISION_SCHEMA_VERSION",
    "LATENT_WORLD_SCHEMA_VERSION",
    "WORLD_INVARIANT_SCHEMA_VERSION",
    "WORLD_RECIPE_SCHEMA_VERSION",
    "FamilyRevisionAuthority",
    "LatentWorldManifest",
    "PolicyDigestBinding",
    "WorldEntityState",
    "WorldInvariant",
    "WorldParameter",
    "WorldRecipe",
]
