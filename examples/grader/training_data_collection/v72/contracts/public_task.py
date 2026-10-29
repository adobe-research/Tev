"""Strict solver-visible task and package contracts for v7.2."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, ValidationInfo, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import (
    JsonValue,
    content_digest,
    content_id,
    sha256_bytes,
)

from .identity import ContentIdentifier, Sha256Digest, StrictFrozenModel
from .world import FamilyRevisionAuthority

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=160,
        pattern=r"^[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*$",
        strict=True,
    ),
]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, strict=True)]

_PRIVATE_PATH_COMPONENTS = frozenset(
    {"private", "private_truth", "hidden_suite", "reference_answer", "verifier", "witness"}
)


def _validate_public_path(value: str) -> str:
    path = safe_relative_path(value)
    if any(part in _PRIVATE_PATH_COMPONENTS for part in path.parts):
        raise ValueError(f"private material cannot be declared in a public path: {value!r}")
    return path.as_posix()


class JSONFieldSpecification(StrictFrozenModel):
    name: NonEmptyString
    value_type: Literal["string", "integer", "number", "boolean", "object", "array"]
    required: bool = True
    nullable: bool = False


class JSONArtifactSchema(StrictFrozenModel):
    kind: Literal["json"] = "json"
    fields: tuple[JSONFieldSpecification, ...] = Field(min_length=1)
    allow_extra_keys: bool = False

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("JSON field names must be unique")
        return self


ScalarType = Literal["string", "integer", "number", "boolean"]


class CSVColumnSpecification(StrictFrozenModel):
    name: NonEmptyString
    scalar_type: ScalarType
    nullable: bool = False


class CSVArtifactSchema(StrictFrozenModel):
    kind: Literal["csv"] = "csv"
    columns: tuple[CSVColumnSpecification, ...] = Field(min_length=1)
    row_key: tuple[NonEmptyString, ...] = ()
    row_order: Literal["unspecified", "input", "sorted_ascending", "sorted_descending"] = (
        "unspecified"
    )

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("CSV column names must be unique")
        if len(self.row_key) != len(set(self.row_key)) or not set(self.row_key).issubset(names):
            raise ValueError("CSV row keys must be unique declared columns")
        return self


SQLiteType = Literal[
    "INTEGER",
    "REAL",
    "TEXT",
    "BLOB",
    "NUMERIC",
    "BOOLEAN",
    "DATE",
    "DATETIME",
]


class SQLiteColumnSpecification(StrictFrozenModel):
    name: NonEmptyString
    sql_type: SQLiteType
    nullable: bool = False


class SQLiteTableSpecification(StrictFrozenModel):
    name: NonEmptyString
    columns: tuple[SQLiteColumnSpecification, ...] = Field(min_length=1)
    primary_key: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("SQLite column names must be unique within a table")
        if len(self.primary_key) != len(set(self.primary_key)) or not set(
            self.primary_key
        ).issubset(names):
            raise ValueError("SQLite primary keys must be unique declared columns")
        return self


class SQLiteArtifactSchema(StrictFrozenModel):
    kind: Literal["sqlite"] = "sqlite"
    tables: tuple[SQLiteTableSpecification, ...] = Field(min_length=1)
    allow_extra_tables: bool = False

    @model_validator(mode="after")
    def validate_tables(self) -> Self:
        names = [table.name for table in self.tables]
        if len(names) != len(set(names)):
            raise ValueError("SQLite table names must be unique")
        return self


class TextArtifactSchema(StrictFrozenModel):
    kind: Literal["text"] = "text"
    format: Literal["markdown", "plain_text"]
    encoding: Literal["utf-8"] = "utf-8"
    required_sections: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_sections(self) -> Self:
        if len(self.required_sections) != len(set(self.required_sections)):
            raise ValueError("text required sections must be unique")
        return self


class BinaryArtifactSchema(StrictFrozenModel):
    kind: Literal["binary"] = "binary"
    media_type: NonEmptyString
    minimum_size_bytes: int = Field(default=1, ge=0)
    maximum_size_bytes: int | None = Field(default=None, ge=0)
    magic_prefix_hex: str | None = Field(default=None, pattern=r"^(?:[0-9a-f]{2})+$")

    @model_validator(mode="after")
    def validate_size_range(self) -> Self:
        if self.maximum_size_bytes is not None and (
            self.maximum_size_bytes < self.minimum_size_bytes
        ):
            raise ValueError("binary maximum size cannot be below its minimum size")
        return self


NativeArtifactSchema = Annotated[
    JSONArtifactSchema
    | CSVArtifactSchema
    | SQLiteArtifactSchema
    | TextArtifactSchema
    | BinaryArtifactSchema,
    Field(discriminator="kind"),
]


class PublicInputRole(StrictFrozenModel):
    role_id: Identifier
    artifact_id: Identifier
    root: Literal["input", "software"] = "input"
    path: str
    path_kind: Literal["file", "directory"]
    artifact_type: NonEmptyString
    description: NonEmptyString
    required: bool = True
    critical: bool = False
    workflow_stage_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        _validate_public_path(self.path)
        if self.critical and not self.required:
            raise ValueError("critical public input roles must be required")
        if len(self.workflow_stage_ids) != len(set(self.workflow_stage_ids)):
            raise ValueError("input workflow stage references must be unique")
        return self


class OutputArtifact(StrictFrozenModel):
    artifact_id: Identifier
    path: str
    purpose: NonEmptyString
    native_schema: NativeArtifactSchema
    public_rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    produced_by_stage_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        _validate_public_path(self.path)
        if len(self.public_rule_ids) != len(set(self.public_rule_ids)):
            raise ValueError("output public rule references must be unique")
        if len(self.produced_by_stage_ids) != len(set(self.produced_by_stage_ids)):
            raise ValueError("output workflow stage references must be unique")
        return self


PublicRelationType = Literal[
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


class PublicArtifactRelation(StrictFrozenModel):
    relation_id: Identifier
    relation_type: PublicRelationType
    source_artifact_ids: tuple[Identifier, ...] = Field(min_length=1)
    target_artifact_ids: tuple[Identifier, ...] = Field(min_length=1)
    description: NonEmptyString
    workflow_stage_ids: tuple[Identifier, ...] = Field(min_length=1)
    public_rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    validator_id: Identifier
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_relation(self) -> Self:
        for values, label in (
            (self.source_artifact_ids, "source artifacts"),
            (self.target_artifact_ids, "target artifacts"),
            (self.workflow_stage_ids, "workflow stages"),
            (self.public_rule_ids, "public rules"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"relation {label} must be unique")
        if set(self.source_artifact_ids) & set(self.target_artifact_ids):
            raise ValueError("relation sources and targets must be distinct")
        return self


class PublicRule(StrictFrozenModel):
    rule_id: Identifier
    description: NonEmptyString
    validator_id: Identifier
    input_role_ids: tuple[Identifier, ...] = ()
    target_artifact_ids: tuple[Identifier, ...] = Field(min_length=1)
    relation_ids: tuple[Identifier, ...] = ()
    workflow_stage_ids: tuple[Identifier, ...] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for values, label in (
            (self.input_role_ids, "input roles"),
            (self.target_artifact_ids, "target artifacts"),
            (self.relation_ids, "relations"),
            (self.workflow_stage_ids, "workflow stages"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"public rule {label} must be unique")
        return self


class AuthorityPrecedenceRule(StrictFrozenModel):
    precedence_id: Identifier
    description: NonEmptyString
    higher_authority_artifact_id: Identifier
    lower_authority_artifact_ids: tuple[Identifier, ...] = Field(min_length=1)
    public_rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    workflow_stage_ids: tuple[Identifier, ...] = Field(min_length=1)
    validator_id: Identifier

    @model_validator(mode="after")
    def validate_precedence(self) -> Self:
        if len(self.lower_authority_artifact_ids) != len(set(self.lower_authority_artifact_ids)):
            raise ValueError("lower-authority artifact references must be unique")
        if self.higher_authority_artifact_id in self.lower_authority_artifact_ids:
            raise ValueError("one artifact cannot have both higher and lower authority")
        if len(self.public_rule_ids) != len(set(self.public_rule_ids)):
            raise ValueError("precedence public rule references must be unique")
        if len(self.workflow_stage_ids) != len(set(self.workflow_stage_ids)):
            raise ValueError("precedence workflow stage references must be unique")
        return self


class TaskRequirement(StrictFrozenModel):
    requirement_id: Identifier
    ordered_step: int = Field(ge=1)
    description: NonEmptyString
    input_role_ids: tuple[Identifier, ...] = Field(min_length=1)
    relation_ids: tuple[Identifier, ...] = Field(min_length=1)
    workflow_stage_ids: tuple[Identifier, ...] = Field(min_length=1)
    output_artifact_ids: tuple[Identifier, ...] = Field(min_length=1)
    public_rule_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for values, label in (
            (self.input_role_ids, "input roles"),
            (self.relation_ids, "relations"),
            (self.workflow_stage_ids, "workflow stages"),
            (self.output_artifact_ids, "output artifacts"),
            (self.public_rule_ids, "public rules"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"requirement {label} must be unique")
        return self


class PublicEnvironment(StrictFrozenModel):
    workspace_root: Literal["."] = "."
    input_root: Literal["input"] = "input"
    output_root: Literal["output"] = "output"
    software_root: Literal["software"] = "software"
    workspace_layout: Literal["machine_contract", "benchmark_native"] = "machine_contract"
    task_specification_path: str = "task_spec.json"
    prompt_path: Literal["prompt.md"] = "prompt.md"
    shell: NonEmptyString = "/bin/sh"
    network_access: Literal[False] = False
    available_tools: tuple[Identifier, ...] = ()
    permitted_dependencies: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        _validate_public_path(self.task_specification_path)
        expected_specification_path = (
            "task_spec.json"
            if self.workspace_layout == "machine_contract"
            else "input/task_prompt.md"
        )
        if self.task_specification_path != expected_specification_path:
            raise ValueError("public task path differs from the selected workspace layout")
        if len(self.available_tools) != len(set(self.available_tools)):
            raise ValueError("available tools must be unique")
        if len(self.permitted_dependencies) != len(set(self.permitted_dependencies)):
            raise ValueError("permitted dependencies must be unique")
        return self


class PublicEvaluation(StrictFrozenModel):
    allow_extra_files: bool = False
    validation_command: tuple[NonEmptyString, ...] = Field(min_length=1)
    public_rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    default_numeric_tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_rules(self) -> Self:
        if len(self.public_rule_ids) != len(set(self.public_rule_ids)):
            raise ValueError("public evaluation rule references must be unique")
        return self


class FrozenAuthoredPrompt(StrictFrozenModel):
    """Offline-authored prompt prose frozen into the public task authority."""

    schema_version: Literal["v7.2.authored-prompt.v1"] = "v7.2.authored-prompt.v1"
    authored_prompt_id: ContentIdentifier
    text: NonEmptyString
    prompt_sha256: Sha256Digest
    model_id: NonEmptyString
    temperature: Literal[0.0] = 0.0
    transcript_sha256: Sha256Digest
    benchmark_corpus_sha256: Sha256Digest
    maximum_benchmark_similarity: float = Field(ge=0, le=1, allow_inf_nan=False)
    observed_benchmark_similarity: float = Field(ge=0, le=1, allow_inf_nan=False)
    fidelity_gate_version: Literal["v72-prompt-fidelity.v1"] = "v72-prompt-fidelity.v1"
    leakage_gate_version: Literal["v72-prompt-leakage.v1"] = "v72-prompt-leakage.v1"
    decontamination_gate_version: Literal["v72-prompt-decontamination.v1"] = (
        "v72-prompt-decontamination.v1"
    )

    @model_validator(mode="after")
    def validate_authored_prompt(self, info: ValidationInfo) -> Self:
        if not self.text.endswith("\n"):
            raise ValueError("authored prompt text must end with one newline")
        if "\x00" in self.text:
            raise ValueError("authored prompt text cannot contain NUL bytes")
        if sha256_bytes(self.text.encode("utf-8")) != self.prompt_sha256:
            raise ValueError("authored prompt digest differs from its UTF-8 bytes")
        if self.observed_benchmark_similarity > self.maximum_benchmark_similarity:
            raise ValueError("authored prompt exceeds its benchmark similarity limit")
        if not (info.context or {}).get("skip_identity") and (
            self.authored_prompt_id != self.expected_authored_prompt_id()
        ):
            raise ValueError("authored_prompt_id does not match frozen prompt authority")
        return self

    def expected_authored_prompt_id(self) -> str:
        return content_id(
            "v72-authored-prompt",
            self.model_dump(mode="json", exclude={"authored_prompt_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"authored_prompt_id": f"v72-authored-prompt:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            authored_prompt_id=provisional.expected_authored_prompt_id(),
            **provisional.model_dump(mode="python", exclude={"authored_prompt_id"}),
        )


class PublicTaskSpecification(StrictFrozenModel):
    """Complete public authority from which the policy prompt is rendered."""

    schema_version: Literal["v7.2.public-task.v1"] = "v7.2.public-task.v1"
    specification_id: ContentIdentifier
    title: NonEmptyString
    objective: NonEmptyString
    scenario: NonEmptyString
    public_inputs: tuple[PublicInputRole, ...] = Field(min_length=1)
    ordered_requirements: tuple[TaskRequirement, ...] = Field(min_length=1)
    output_artifacts: tuple[OutputArtifact, ...] = Field(min_length=1)
    public_rules: tuple[PublicRule, ...] = Field(min_length=1)
    public_artifact_relations: tuple[PublicArtifactRelation, ...] = Field(min_length=1)
    authority_and_precedence_rules: tuple[AuthorityPrecedenceRule, ...] = Field(min_length=1)
    environment: PublicEnvironment
    public_evaluation: PublicEvaluation
    authored_prompt: FrozenAuthoredPrompt | None = None

    @model_validator(mode="after")
    def validate_specification(self, info: ValidationInfo) -> Self:
        collections = (
            ([item.role_id for item in self.public_inputs], "public input role IDs"),
            ([item.artifact_id for item in self.public_inputs], "public input artifact IDs"),
            (
                [f"{item.root}/{item.path}" for item in self.public_inputs],
                "public input paths",
            ),
            (
                [item.requirement_id for item in self.ordered_requirements],
                "requirement IDs",
            ),
            ([item.artifact_id for item in self.output_artifacts], "output artifact IDs"),
            ([item.path for item in self.output_artifacts], "output paths"),
            ([item.rule_id for item in self.public_rules], "public rule IDs"),
            (
                [item.relation_id for item in self.public_artifact_relations],
                "public relation IDs",
            ),
            (
                [item.precedence_id for item in self.authority_and_precedence_rules],
                "precedence IDs",
            ),
        )
        for values, label in collections:
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        input_paths = [
            (PurePosixPath(item.root) / item.path, item.path_kind) for item in self.public_inputs
        ]
        for index, (path, kind) in enumerate(input_paths):
            for other_path, other_kind in input_paths[index + 1 :]:
                if (kind == "directory" and other_path.is_relative_to(path)) or (
                    other_path == path
                ):
                    raise ValueError("public input roles cannot overlap")
                if other_kind == "directory" and path.is_relative_to(other_path):
                    raise ValueError("public input roles cannot overlap")
        if sorted(item.ordered_step for item in self.ordered_requirements) != list(
            range(1, len(self.ordered_requirements) + 1)
        ):
            raise ValueError("requirement steps must be contiguous from one")

        role_ids = {item.role_id for item in self.public_inputs}
        input_artifacts = {item.artifact_id: item.role_id for item in self.public_inputs}
        output_artifacts = {item.artifact_id for item in self.output_artifacts}
        all_artifacts = set(input_artifacts) | output_artifacts
        rules = {item.rule_id: item for item in self.public_rules}
        relations = {item.relation_id: item for item in self.public_artifact_relations}
        stages = {
            stage_id for item in self.public_inputs for stage_id in item.workflow_stage_ids
        } | {stage_id for item in self.output_artifacts for stage_id in item.produced_by_stage_ids}

        for relation in self.public_artifact_relations:
            if not (set(relation.source_artifact_ids) | set(relation.target_artifact_ids)).issubset(
                all_artifacts
            ):
                raise ValueError("public relations must reference declared artifacts")
            if not set(relation.public_rule_ids).issubset(rules):
                raise ValueError("public relations must reference declared rules")
            expected_rules = {
                rule.rule_id
                for rule in self.public_rules
                if relation.relation_id in rule.relation_ids
            }
            if set(relation.public_rule_ids) != expected_rules:
                raise ValueError("relation rule references must exactly match executable rules")
            stages.update(relation.workflow_stage_ids)
        for rule in self.public_rules:
            if not set(rule.input_role_ids).issubset(role_ids):
                raise ValueError("public rules must reference declared input roles")
            if not set(rule.target_artifact_ids).issubset(output_artifacts):
                raise ValueError("public rules must target declared output artifacts")
            if not set(rule.relation_ids).issubset(relations):
                raise ValueError("public rules must reference declared relations")
            stages.update(rule.workflow_stage_ids)
        for precedence in self.authority_and_precedence_rules:
            referenced = {
                precedence.higher_authority_artifact_id,
                *precedence.lower_authority_artifact_ids,
            }
            if not referenced.issubset(all_artifacts):
                raise ValueError("precedence rules must reference declared artifacts")
            if not set(precedence.public_rule_ids).issubset(rules):
                raise ValueError("precedence rules must reference declared public rules")
            for rule_id in precedence.public_rule_ids:
                rule = rules[rule_id]
                rule_artifacts = {
                    *(
                        item.artifact_id
                        for item in self.public_inputs
                        if item.role_id in rule.input_role_ids
                    ),
                    *rule.target_artifact_ids,
                }
                if not rule_artifacts & referenced:
                    raise ValueError("precedence rules cannot bind unrelated public rules")
            stages.update(precedence.workflow_stage_ids)

        referenced_roles: set[str] = set()
        referenced_outputs: set[str] = set()
        referenced_relations: set[str] = set()
        referenced_rules: set[str] = set()
        for requirement in self.ordered_requirements:
            if not set(requirement.input_role_ids).issubset(role_ids):
                raise ValueError("requirements must reference declared input roles")
            if not set(requirement.output_artifact_ids).issubset(output_artifacts):
                raise ValueError("requirements must reference declared output artifacts")
            if not set(requirement.relation_ids).issubset(relations):
                raise ValueError("requirements must reference declared relations")
            if not set(requirement.public_rule_ids).issubset(rules):
                raise ValueError("requirements must reference declared public rules")
            if not set(requirement.workflow_stage_ids).issubset(stages):
                raise ValueError("requirements must reference declared workflow stages")

            requirement_relations = [relations[item] for item in requirement.relation_ids]
            relation_artifacts = {
                artifact_id
                for relation in requirement_relations
                for artifact_id in (*relation.source_artifact_ids, *relation.target_artifact_ids)
            }
            requirement_rules = [rules[item] for item in requirement.public_rule_ids]
            derived_roles = {
                input_artifacts[artifact_id]
                for artifact_id in relation_artifacts
                if artifact_id in input_artifacts
            } | {role_id for rule in requirement_rules for role_id in rule.input_role_ids}
            derived_outputs = (relation_artifacts & output_artifacts) | {
                artifact_id
                for rule in requirement_rules
                for artifact_id in rule.target_artifact_ids
            }
            derived_stages = {
                stage_id
                for relation in requirement_relations
                for stage_id in relation.workflow_stage_ids
            } | {stage_id for rule in requirement_rules for stage_id in rule.workflow_stage_ids}
            if (
                set(requirement.input_role_ids) != derived_roles
                or set(requirement.output_artifact_ids) != derived_outputs
                or set(requirement.workflow_stage_ids) != derived_stages
            ):
                raise ValueError(
                    "requirement references must exactly match their rules and relations"
                )
            if any(
                not set(rule.input_role_ids).issubset(requirement.input_role_ids)
                or not set(rule.target_artifact_ids).issubset(requirement.output_artifact_ids)
                or not set(rule.relation_ids).issubset(requirement.relation_ids)
                or not set(rule.workflow_stage_ids).issubset(requirement.workflow_stage_ids)
                for rule in requirement_rules
            ):
                raise ValueError("requirements cannot bind unrelated public rules")
            referenced_roles.update(requirement.input_role_ids)
            referenced_outputs.update(requirement.output_artifact_ids)
            referenced_relations.update(requirement.relation_ids)
            referenced_rules.update(requirement.public_rule_ids)

        if referenced_roles != role_ids or referenced_outputs != output_artifacts:
            raise ValueError("every public input and output must have an exact requirement")
        rule_relation_references = {
            relation_id for rule in self.public_rules for relation_id in rule.relation_ids
        }
        if referenced_relations | rule_relation_references != set(relations):
            raise ValueError("every public relation must be bound to a requirement or rule")
        output_rule_references = {
            rule_id for output in self.output_artifacts for rule_id in output.public_rule_ids
        }
        if referenced_rules | output_rule_references | set(
            self.public_evaluation.public_rule_ids
        ) != set(rules):
            raise ValueError("every public rule must be bound to requirements and evaluation")
        if set(self.public_evaluation.public_rule_ids) != set(rules):
            raise ValueError("public evaluation must execute every public rule")
        for output in self.output_artifacts:
            if not set(output.public_rule_ids).issubset(rules):
                raise ValueError("outputs must reference declared public rules")
            expected_rules = {
                rule.rule_id
                for rule in self.public_rules
                if output.artifact_id in rule.target_artifact_ids
            }
            if set(output.public_rule_ids) != expected_rules:
                raise ValueError("output rule references must exactly match executable rules")
            if not set(output.produced_by_stage_ids).issubset(stages):
                raise ValueError("outputs must reference declared workflow stages")
        if not (info.context or {}).get("skip_identity") and (
            self.specification_id != self.expected_specification_id()
        ):
            raise ValueError("specification_id does not match public task authority")
        return self

    def expected_specification_id(self) -> str:
        authority = self.model_dump(
            mode="json",
            exclude={"specification_id", "authored_prompt"},
        )
        if self.authored_prompt is not None:
            authority["authored_prompt"] = self.authored_prompt.model_dump(mode="json")
        return content_id(
            "v72-public-task",
            authority,
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"specification_id": f"v72-public-task:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            specification_id=provisional.expected_specification_id(),
            **provisional.model_dump(mode="python", exclude={"specification_id"}),
        )


class ManifestEntry(StrictFrozenModel):
    path: str
    visibility: Literal["public", "private"]
    sha256: Sha256Digest
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        safe_relative_path(self.path)
        return self


class FileManifest(StrictFrozenModel):
    schema_version: Literal["v7.2.file-manifest.v1"] = "v7.2.file-manifest.v1"
    manifest_id: ContentIdentifier
    visibility: Literal["public", "private"]
    tree_sha256: Sha256Digest
    entries: tuple[ManifestEntry, ...]

    @model_validator(mode="after")
    def validate_manifest(self, info: ValidationInfo) -> Self:
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("file manifest paths must be sorted and unique")
        if any(entry.visibility != self.visibility for entry in self.entries):
            raise ValueError("file manifest entries must match manifest visibility")
        tree_authority = [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in self.entries
        ]
        if self.tree_sha256 != content_digest(tree_authority):
            raise ValueError("file manifest tree digest does not match its entries")
        if not (info.context or {}).get("skip_identity") and (
            self.manifest_id != self.expected_manifest_id()
        ):
            raise ValueError("manifest_id does not match file manifest authority")
        return self

    def expected_manifest_id(self) -> str:
        return content_id(
            "v72-file-manifest",
            self.model_dump(mode="json", exclude={"manifest_id"}),
        )

    @classmethod
    def identified(
        cls,
        *,
        visibility: Literal["public", "private"],
        tree_sha256: str,
        entries: tuple[ManifestEntry, ...],
    ) -> Self:
        values = {"visibility": visibility, "tree_sha256": tree_sha256, "entries": entries}
        provisional = cls.model_validate(
            {"manifest_id": f"v72-file-manifest:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(manifest_id=provisional.expected_manifest_id(), **values)


class TaskPackageManifest(StrictFrozenModel):
    """Content identity for every package authority needed by later release phases."""

    schema_version: Literal["v7.2.task.v1"] = "v7.2.task.v1"
    task_id: ContentIdentifier
    family_revision: FamilyRevisionAuthority
    recipe_id: ContentIdentifier
    world_id: ContentIdentifier
    artifact_graph_id: ContentIdentifier
    workflow_id: ContentIdentifier
    corruption_manifest_id: ContentIdentifier
    hidden_suite_id: ContentIdentifier
    public_specification_id: ContentIdentifier
    prompt_sha256: Sha256Digest
    public_manifest_id: ContentIdentifier
    private_manifest_id: ContentIdentifier
    public_tree_sha256: Sha256Digest
    private_tree_sha256: Sha256Digest
    public_input_tree_sha256: Sha256Digest
    scale_id: NonEmptyString
    curriculum_layer: Literal["primitive_terminal", "family_integration", "professional_frontier"]
    behavior_seed: int = Field(ge=0, lt=2**128)
    source_record_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    policy_digest_set_id: ContentIdentifier
    reward_contract_id: ContentIdentifier | None = None
    progress_contract_id: ContentIdentifier | None = None

    @model_validator(mode="after")
    def validate_identity(self, info: ValidationInfo) -> Self:
        if self.source_record_ids != tuple(sorted(set(self.source_record_ids))):
            raise ValueError("task package source record identities must be sorted and unique")
        if not (info.context or {}).get("skip_identity") and (
            self.task_id != self.expected_task_id()
        ):
            raise ValueError("task_id does not match package authority")
        return self

    def expected_task_id(self) -> str:
        return content_id("task", self.model_dump(mode="json", exclude={"task_id"}))

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"task_id": f"task:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            task_id=provisional.expected_task_id(),
            **provisional.model_dump(mode="python", exclude={"task_id"}),
        )
