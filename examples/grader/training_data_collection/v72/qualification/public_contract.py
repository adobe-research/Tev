"""Executable validation of v7.2 solver-visible task contracts."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import ValidationInfo, model_validator

from training_data_collection.common.filesystem import (
    iter_regular_files,
    resolve_under,
    tree_sha256,
)
from training_data_collection.common.identity import content_digest, content_id
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)
from training_data_collection.v72.contracts.public_task import (
    AuthorityPrecedenceRule,
    BinaryArtifactSchema,
    CSVArtifactSchema,
    JSONArtifactSchema,
    OutputArtifact,
    PublicArtifactRelation,
    PublicInputRole,
    PublicRule,
    PublicTaskSpecification,
    SQLiteArtifactSchema,
    TextArtifactSchema,
)

PublicValidationTarget = PublicRule | PublicArtifactRelation | AuthorityPrecedenceRule


class PublicRuleViolation(ValueError):
    """Raised by a public validator when realized artifacts violate its published rule."""


@dataclass(frozen=True)
class PublicValidatorContext:
    specification: PublicTaskSpecification
    input_root: Path
    output_root: Path
    target_kind: Literal["rule", "relation", "precedence"]
    target: PublicValidationTarget


type PublicValidator = Callable[[PublicValidatorContext], None]


class PublicValidatorExecution(StrictFrozenModel):
    target_kind: Literal[
        "input_coverage",
        "output_coverage",
        "native_schema",
        "rule",
        "relation",
        "precedence",
        "validator_isolation",
    ]
    target_id: str
    validator_id: str
    passed: bool
    detail: str = ""


class PublicContractEvidence(StrictFrozenModel):
    """Content-bound result of executing every public structural and behavioral validator."""

    schema_version: Literal["v7.2.public-contract-evidence.v1"] = "v7.2.public-contract-evidence.v1"
    evidence_id: ContentIdentifier
    specification_id: ContentIdentifier
    input_tree_sha256: Sha256Digest
    output_tree_sha256: Sha256Digest
    executions: tuple[PublicValidatorExecution, ...]
    passed: bool

    @model_validator(mode="after")
    def validate_evidence(self, info: ValidationInfo) -> Self:
        if not self.executions:
            raise ValueError("public contract evidence requires validator executions")
        if self.passed != all(execution.passed for execution in self.executions):
            raise ValueError("public contract result must be derived from all validator executions")
        if not (info.context or {}).get("skip_identity") and (
            self.evidence_id != self.expected_evidence_id()
        ):
            raise ValueError("evidence_id does not match public contract evidence")
        return self

    def expected_evidence_id(self) -> str:
        return content_id(
            "v72-public-contract-evidence",
            self.model_dump(mode="json", exclude={"evidence_id"}),
        )

    @classmethod
    def identified(
        cls,
        *,
        specification_id: str,
        input_tree_sha256: str,
        output_tree_sha256: str,
        executions: tuple[PublicValidatorExecution, ...],
    ) -> Self:
        values = {
            "specification_id": specification_id,
            "input_tree_sha256": input_tree_sha256,
            "output_tree_sha256": output_tree_sha256,
            "executions": executions,
            "passed": all(execution.passed for execution in executions),
        }
        provisional = cls.model_validate(
            {"evidence_id": f"v72-public-contract-evidence:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(evidence_id=provisional.expected_evidence_id(), **values)


def _role_matches_file(role: PublicInputRole, relative_path: str) -> bool:
    role_path = PurePosixPath(role.path)
    candidate = PurePosixPath(relative_path)
    return (
        candidate == role_path if role.path_kind == "file" else candidate.is_relative_to(role_path)
    )


def _input_coverage_issues(
    specification: PublicTaskSpecification,
    input_root: Path,
) -> list[str]:
    issues: list[str] = []
    roots = {"input": input_root}
    if any(role.root == "software" for role in specification.public_inputs):
        roots["software"] = input_root.parent / "software"
    try:
        files_by_root = {name: iter_regular_files(root) for name, root in roots.items()}
    except ValueError as error:
        return [str(error)]
    for role in specification.public_inputs:
        try:
            target = resolve_under(roots[role.root], role.path)
        except ValueError as error:
            issues.append(str(error))
            continue
        if role.required and not target.exists():
            issues.append(f"required input role is missing: {role.role_id}")
        elif target.exists() and role.path_kind == "file" and not target.is_file():
            issues.append(f"input role is not a file: {role.role_id}")
        elif target.exists() and role.path_kind == "directory" and not target.is_dir():
            issues.append(f"input role is not a directory: {role.role_id}")
    for root_name, files in files_by_root.items():
        for path in files:
            relative = path.relative_to(roots[root_name]).as_posix()
            if (
                specification.environment.workspace_layout == "benchmark_native"
                and root_name == "input"
                and relative == "task_prompt.md"
            ):
                continue
            matches = [
                role.role_id
                for role in specification.public_inputs
                if role.root == root_name and _role_matches_file(role, relative)
            ]
            if len(matches) != 1:
                issues.append(
                    f"input file must match exactly one public role: {root_name}/{relative}"
                )
        role_paths = [
            PurePosixPath(role.path)
            for role in specification.public_inputs
            if role.root == root_name
        ]
        for directory in sorted(path for path in roots[root_name].rglob("*") if path.is_dir()):
            relative = PurePosixPath(directory.relative_to(roots[root_name]).as_posix())
            if not any(
                relative == role_path
                or relative.is_relative_to(role_path)
                or role_path.is_relative_to(relative)
                for role_path in role_paths
            ):
                issues.append(f"undeclared input directory: {root_name}/{relative.as_posix()}")
    return issues


def _input_authority_digest(
    specification: PublicTaskSpecification,
    input_root: Path,
) -> str:
    if specification.environment.workspace_layout == "machine_contract":
        return tree_sha256(input_root)
    return content_digest(
        {
            "input": tree_sha256(input_root),
            "software": tree_sha256(input_root.parent / "software"),
        }
    )


def _output_coverage_issues(
    specification: PublicTaskSpecification,
    output_root: Path,
) -> list[str]:
    try:
        files = iter_regular_files(output_root)
    except ValueError as error:
        return [str(error)]
    actual = {path.relative_to(output_root).as_posix() for path in files}
    expected = {artifact.path for artifact in specification.output_artifacts}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    issues = [f"required output is missing: {path}" for path in missing]
    if not specification.public_evaluation.allow_extra_files:
        issues.extend(f"undeclared output is present: {path}" for path in extra)
    return issues


def _json_value_matches(value: object, expected: str) -> bool:
    if expected == "string":
        return type(value) is str
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in (int, float) and math.isfinite(value)
    if expected == "boolean":
        return type(value) is bool
    if expected == "object":
        return type(value) is dict
    return type(value) is list


def _validate_json(path: Path, schema: JSONArtifactSchema) -> None:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRuleViolation(f"invalid JSON artifact: {type(error).__name__}") from error
    if type(value) is not dict:
        raise PublicRuleViolation("JSON output must be an object")
    fields = {field.name: field for field in schema.fields}
    missing = sorted(
        field.name for field in schema.fields if field.required and field.name not in value
    )
    if missing:
        raise PublicRuleViolation(f"JSON output is missing required keys: {missing}")
    if not schema.allow_extra_keys and set(value) - set(fields):
        raise PublicRuleViolation(
            f"JSON output has undeclared keys: {sorted(set(value) - set(fields))}"
        )
    for name, item in value.items():
        field = fields.get(name)
        if field is None:
            continue
        if item is None:
            if not field.nullable:
                raise PublicRuleViolation(f"JSON key is not nullable: {name}")
        elif not _json_value_matches(item, field.value_type):
            raise PublicRuleViolation(f"JSON key has the wrong type: {name}")


def _validate_csv_scalar(value: str, expected: str, *, nullable: bool) -> bool:
    if not value:
        return nullable
    if expected == "string":
        return True
    if expected == "integer":
        try:
            int(value)
        except ValueError:
            return False
        return not any(character in value.lower() for character in (".", "e"))
    if expected == "number":
        try:
            return math.isfinite(float(value))
        except ValueError:
            return False
    return value.lower() in {"true", "false", "0", "1"}


def _validate_csv(path: Path, schema: CSVArtifactSchema) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            expected_names = [column.name for column in schema.columns]
            if reader.fieldnames != expected_names:
                raise PublicRuleViolation("CSV header does not exactly match declared columns")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise PublicRuleViolation(f"invalid CSV artifact: {type(error).__name__}") from error
    for row_index, row in enumerate(rows, start=2):
        for column in schema.columns:
            value = row.get(column.name)
            if value is None or not _validate_csv_scalar(
                value,
                column.scalar_type,
                nullable=column.nullable,
            ):
                raise PublicRuleViolation(
                    f"CSV value violates {column.scalar_type} at row {row_index}, "
                    f"column {column.name}"
                )
    if schema.row_key:
        keys = [tuple(row[name] for name in schema.row_key) for row in rows]
        if len(keys) != len(set(keys)):
            raise PublicRuleViolation("CSV row key is not unique")
        if schema.row_order == "sorted_ascending" and keys != sorted(keys):
            raise PublicRuleViolation("CSV rows are not sorted ascending by the row key")
        if schema.row_order == "sorted_descending" and keys != sorted(keys, reverse=True):
            raise PublicRuleViolation("CSV rows are not sorted descending by the row key")


def _validate_sqlite(path: Path, schema: SQLiteArtifactSchema) -> None:
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise PublicRuleViolation(f"invalid SQLite artifact: {type(error).__name__}") from error
    try:
        actual_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected_tables = {table.name for table in schema.tables}
        if not expected_tables.issubset(actual_tables):
            raise PublicRuleViolation(
                f"SQLite output is missing tables: {sorted(expected_tables - actual_tables)}"
            )
        if not schema.allow_extra_tables and actual_tables != expected_tables:
            raise PublicRuleViolation(
                f"SQLite output has undeclared tables: {sorted(actual_tables - expected_tables)}"
            )
        for table in schema.tables:
            escaped_name = table.name.replace('"', '""')
            rows = list(connection.execute(f'PRAGMA table_info("{escaped_name}")'))
            actual_names = [row[1] for row in rows]
            expected_names = [column.name for column in table.columns]
            if actual_names != expected_names:
                raise PublicRuleViolation(
                    f"SQLite columns differ for table {table.name}: {actual_names}"
                )
            primary_key = tuple(
                row[1] for row in sorted(rows, key=lambda item: item[5]) if row[5] > 0
            )
            if primary_key != table.primary_key:
                raise PublicRuleViolation(f"SQLite primary key differs for table {table.name}")
            for row, column in zip(rows, table.columns, strict=True):
                if row[2].upper() != column.sql_type:
                    raise PublicRuleViolation(f"SQLite type differs for {table.name}.{column.name}")
                if not column.nullable and not row[3] and row[5] == 0:
                    raise PublicRuleViolation(
                        f"SQLite NOT NULL constraint is missing for {table.name}.{column.name}"
                    )
    except sqlite3.Error as error:
        raise PublicRuleViolation(f"SQLite validation failed: {type(error).__name__}") from error
    finally:
        connection.close()


def _validate_text(path: Path, schema: TextArtifactSchema) -> None:
    try:
        value = path.read_text(encoding=schema.encoding)
    except (OSError, UnicodeDecodeError) as error:
        raise PublicRuleViolation(f"invalid text artifact: {type(error).__name__}") from error
    missing = [section for section in schema.required_sections if section not in value]
    if missing:
        raise PublicRuleViolation(f"text artifact is missing required sections: {missing}")


def _validate_binary(path: Path, schema: BinaryArtifactSchema) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(len(schema.magic_prefix_hex or "") // 2)
    except OSError as error:
        raise PublicRuleViolation(f"invalid binary artifact: {type(error).__name__}") from error
    if size < schema.minimum_size_bytes:
        raise PublicRuleViolation("binary artifact is below its minimum size")
    if schema.maximum_size_bytes is not None and size > schema.maximum_size_bytes:
        raise PublicRuleViolation("binary artifact exceeds its maximum size")
    if schema.magic_prefix_hex is not None and prefix.hex() != schema.magic_prefix_hex:
        raise PublicRuleViolation("binary artifact has the wrong magic prefix")


def _validate_native_artifact(output_root: Path, artifact: OutputArtifact) -> None:
    path = resolve_under(output_root, artifact.path)
    if path.is_symlink() or not path.is_file():
        raise PublicRuleViolation(f"required output is not a regular file: {artifact.path}")
    schema = artifact.native_schema
    if isinstance(schema, JSONArtifactSchema):
        _validate_json(path, schema)
    elif isinstance(schema, CSVArtifactSchema):
        _validate_csv(path, schema)
    elif isinstance(schema, SQLiteArtifactSchema):
        _validate_sqlite(path, schema)
    elif isinstance(schema, TextArtifactSchema):
        _validate_text(path, schema)
    else:
        _validate_binary(path, schema)


def _execution(
    *,
    target_kind: Literal[
        "input_coverage",
        "output_coverage",
        "native_schema",
        "rule",
        "relation",
        "precedence",
        "validator_isolation",
    ],
    target_id: str,
    validator_id: str,
    action: Callable[[], None],
) -> PublicValidatorExecution:
    try:
        action()
    # Validators are pluggable task code. Their failures must become fail-closed evidence.
    except Exception as error:  # noqa: BLE001
        return PublicValidatorExecution(
            target_kind=target_kind,
            target_id=target_id,
            validator_id=validator_id,
            passed=False,
            detail=f"{type(error).__name__}: {error}",
        )
    return PublicValidatorExecution(
        target_kind=target_kind,
        target_id=target_id,
        validator_id=validator_id,
        passed=True,
    )


def validate_public_contract(
    specification: PublicTaskSpecification,
    input_root: Path,
    output_root: Path,
    validators: Mapping[str, PublicValidator],
) -> PublicContractEvidence:
    """Execute native, rule, relation, and precedence validators against realized files."""
    input_before = _input_authority_digest(specification, input_root)
    output_before = tree_sha256(output_root)
    executions: list[PublicValidatorExecution] = []

    input_issues = _input_coverage_issues(specification, input_root)
    executions.append(
        PublicValidatorExecution(
            target_kind="input_coverage",
            target_id="public_inputs",
            validator_id="v72.public.input-coverage.v1",
            passed=not input_issues,
            detail="; ".join(input_issues),
        )
    )
    output_issues = _output_coverage_issues(specification, output_root)
    executions.append(
        PublicValidatorExecution(
            target_kind="output_coverage",
            target_id="output_artifacts",
            validator_id="v72.public.output-coverage.v1",
            passed=not output_issues,
            detail="; ".join(output_issues),
        )
    )
    for artifact in specification.output_artifacts:
        executions.append(
            _execution(
                target_kind="native_schema",
                target_id=artifact.artifact_id,
                validator_id=f"v72.public.native-{artifact.native_schema.kind}.v1",
                action=lambda artifact=artifact: _validate_native_artifact(output_root, artifact),
            )
        )

    targets: tuple[
        tuple[Literal["rule", "relation", "precedence"], PublicValidationTarget], ...
    ] = (
        *(("rule", item) for item in specification.public_rules),
        *(("relation", item) for item in specification.public_artifact_relations),
        *(("precedence", item) for item in specification.authority_and_precedence_rules),
    )
    for target_kind, target in targets:
        validator = validators.get(target.validator_id)
        target_id = (
            target.rule_id
            if isinstance(target, PublicRule)
            else target.relation_id
            if isinstance(target, PublicArtifactRelation)
            else target.precedence_id
        )
        if validator is None:
            executions.append(
                PublicValidatorExecution(
                    target_kind=target_kind,
                    target_id=target_id,
                    validator_id=target.validator_id,
                    passed=False,
                    detail="public validator ID is not registered",
                )
            )
            continue
        context = PublicValidatorContext(
            specification=specification,
            input_root=input_root,
            output_root=output_root,
            target_kind=target_kind,
            target=target,
        )
        executions.append(
            _execution(
                target_kind=target_kind,
                target_id=target_id,
                validator_id=target.validator_id,
                action=lambda validator=validator, context=context: validator(context),
            )
        )

    input_after = _input_authority_digest(specification, input_root)
    output_after = tree_sha256(output_root)
    executions.append(
        PublicValidatorExecution(
            target_kind="validator_isolation",
            target_id="realized_trees",
            validator_id="v72.public.validator-isolation.v1",
            passed=input_before == input_after and output_before == output_after,
            detail=(
                ""
                if input_before == input_after and output_before == output_after
                else "a public validator mutated realized input or output"
            ),
        )
    )
    return PublicContractEvidence.identified(
        specification_id=specification.specification_id,
        input_tree_sha256=input_after,
        output_tree_sha256=output_after,
        executions=tuple(executions),
    )
