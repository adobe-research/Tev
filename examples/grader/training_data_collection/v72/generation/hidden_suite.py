"""Content-bound private hidden-suite contracts and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from training_data_collection.common.filesystem import (
    file_sha256,
    iter_regular_files,
    resolve_under,
    safe_relative_path,
    tree_sha256,
)
from training_data_collection.common.identity import content_digest, content_id
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)


class _HiddenCaseAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.hidden-case.v1"] = "v7.2.hidden-case.v1"
    case_name: str = Field(min_length=1)
    hidden_seed: int = Field(ge=0, lt=2**128)
    file_paths: tuple[str, ...] = Field(min_length=1)
    case_tree_sha256: Sha256Digest
    public_rule_ids: tuple[str, ...] = Field(min_length=1)
    relation_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    stage_ids: tuple[str, ...] = Field(min_length=1)
    reward_component_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_authority(self) -> Self:
        if not self.case_name.strip():
            raise ValueError("hidden case name cannot be blank")
        for path in self.file_paths:
            safe_relative_path(path)
        for label, values in (
            ("hidden case paths", self.file_paths),
            ("hidden case public rules", self.public_rule_ids),
            ("hidden case relations", self.relation_ids),
            ("hidden case stages", self.stage_ids),
            ("hidden case reward components", self.reward_component_ids),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        return self


class HiddenCaseRecord(_HiddenCaseAuthority):
    """One materialized hidden case and its exact semantic coverage."""

    case_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_case_id(self) -> Self:
        if self.case_id != self.expected_case_id():
            raise ValueError("case_id does not match hidden case authority")
        return self

    def expected_case_id(self) -> str:
        return content_id(
            "v72-hidden-case",
            self.model_dump(mode="json", exclude={"case_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> HiddenCaseRecord:
        authority = _HiddenCaseAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            case_id=content_id("v72-hidden-case", authority.model_dump(mode="json")),
        )


class _MetamorphicTransformAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.metamorphic-transform.v1"] = "v7.2.metamorphic-transform.v1"
    operator_id: str = Field(min_length=1)
    source_case_id: ContentIdentifier
    transformed_case_id: ContentIdentifier
    expected_effect: Literal["semantic_change", "semantic_invariance"]
    covered_public_rule_ids: tuple[str, ...] = Field(min_length=1)
    covered_reward_component_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_transform_authority(self) -> Self:
        if not self.operator_id.strip():
            raise ValueError("metamorphic operator identity cannot be blank")
        if self.source_case_id == self.transformed_case_id:
            raise ValueError("metamorphic transform requires distinct cases")
        for label, values in (
            ("metamorphic public rules", self.covered_public_rule_ids),
            ("metamorphic reward components", self.covered_reward_component_ids),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        return self


class MetamorphicTransformRecord(_MetamorphicTransformAuthority):
    """One content-bound relation between two materialized hidden cases."""

    transform_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_transform_id(self) -> Self:
        if self.transform_id != self.expected_transform_id():
            raise ValueError("transform_id does not match metamorphic authority")
        return self

    def expected_transform_id(self) -> str:
        return content_id(
            "v72-metamorphic-transform",
            self.model_dump(mode="json", exclude={"transform_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> MetamorphicTransformRecord:
        authority = _MetamorphicTransformAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            transform_id=content_id("v72-metamorphic-transform", authority.model_dump(mode="json")),
        )


class _HiddenSuiteAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.hidden-suite.v1"] = "v7.2.hidden-suite.v1"
    world_id: ContentIdentifier
    case_records: tuple[HiddenCaseRecord, ...] = Field(min_length=1)
    metamorphic_transform_records: tuple[MetamorphicTransformRecord, ...] = Field(min_length=1)
    public_rule_coverage: tuple[str, ...] = Field(min_length=1)
    relation_coverage: tuple[ContentIdentifier, ...] = Field(min_length=1)
    stage_coverage: tuple[str, ...] = Field(min_length=1)
    reward_component_coverage: tuple[str, ...] = Field(min_length=1)
    hidden_tree_sha256: Sha256Digest
    generator_implementation_digest: Sha256Digest

    @model_validator(mode="after")
    def validate_suite_authority(self) -> Self:
        case_ids = tuple(case.case_id for case in self.case_records)
        case_names = tuple(case.case_name for case in self.case_records)
        hidden_seeds = tuple(case.hidden_seed for case in self.case_records)
        transform_ids = tuple(
            transform.transform_id for transform in self.metamorphic_transform_records
        )
        for label, values in (
            ("hidden case identities", case_ids),
            ("metamorphic transform identities", transform_ids),
            ("hidden public rule coverage", self.public_rule_coverage),
            ("hidden relation coverage", self.relation_coverage),
            ("hidden stage coverage", self.stage_coverage),
            ("hidden reward coverage", self.reward_component_coverage),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if len(case_names) != len(set(case_names)):
            raise ValueError("hidden case names must be unique")
        if len(hidden_seeds) != len(set(hidden_seeds)):
            raise ValueError("hidden case seeds must be unique")
        expected_coverage = (
            (
                self.public_rule_coverage,
                {item for case in self.case_records for item in case.public_rule_ids},
                "public rule",
            ),
            (
                self.relation_coverage,
                {item for case in self.case_records for item in case.relation_ids},
                "relation",
            ),
            (
                self.stage_coverage,
                {item for case in self.case_records for item in case.stage_ids},
                "stage",
            ),
            (
                self.reward_component_coverage,
                {item for case in self.case_records for item in case.reward_component_ids},
                "reward component",
            ),
        )
        for declared, realized, label in expected_coverage:
            if set(declared) != realized:
                raise ValueError(f"hidden suite {label} coverage must equal realized cases")
        case_id_set = set(case_ids)
        if any(
            transform.source_case_id not in case_id_set
            or transform.transformed_case_id not in case_id_set
            for transform in self.metamorphic_transform_records
        ):
            raise ValueError("metamorphic transform references an unknown hidden case")
        if any(
            not set(transform.covered_public_rule_ids).issubset(self.public_rule_coverage)
            or not set(transform.covered_reward_component_ids).issubset(
                self.reward_component_coverage
            )
            for transform in self.metamorphic_transform_records
        ):
            raise ValueError("metamorphic transform coverage exceeds hidden-suite coverage")
        return self


class HiddenSuiteManifest(_HiddenSuiteAuthority):
    """Nonempty private cases, transformations, and semantic coverage authority."""

    hidden_suite_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_hidden_suite_id(self) -> Self:
        if self.hidden_suite_id != self.expected_hidden_suite_id():
            raise ValueError("hidden_suite_id does not match hidden suite authority")
        return self

    def expected_hidden_suite_id(self) -> str:
        return content_id(
            "v72-hidden-suite",
            self.model_dump(mode="json", exclude={"hidden_suite_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> HiddenSuiteManifest:
        authority = _HiddenSuiteAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            hidden_suite_id=content_id("v72-hidden-suite", authority.model_dump(mode="json")),
        )


def hidden_case_tree_sha256(hidden_root: Path, file_paths: tuple[str, ...]) -> str:
    """Hash one hidden case from exact realized file paths."""
    records = []
    for relative in sorted(file_paths):
        path = resolve_under(hidden_root, relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"hidden case path is not a regular file: {relative}")
        records.append({"path": relative, "sha256": file_sha256(path)})
    return content_digest(records)


def validate_hidden_suite(hidden_root: Path, manifest: HiddenSuiteManifest) -> None:
    """Recompute every hidden case and the complete private hidden tree."""
    files = {path.relative_to(hidden_root).as_posix() for path in iter_regular_files(hidden_root)}
    declared_files = [path for case in manifest.case_records for path in case.file_paths]
    covered_files = set(declared_files)
    if len(declared_files) != len(covered_files):
        raise ValueError("hidden-suite files cannot be shared across cases")
    if not files or files != covered_files:
        raise ValueError("hidden cases must cover every nonempty hidden-suite file exactly")
    for case in manifest.case_records:
        if hidden_case_tree_sha256(hidden_root, case.file_paths) != case.case_tree_sha256:
            raise ValueError(f"hidden case tree digest is stale: {case.case_id}")
    if tree_sha256(hidden_root) != manifest.hidden_tree_sha256:
        raise ValueError("hidden suite tree digest is stale")


__all__ = [
    "HiddenCaseRecord",
    "HiddenSuiteManifest",
    "MetamorphicTransformRecord",
    "hidden_case_tree_sha256",
    "validate_hidden_suite",
]
