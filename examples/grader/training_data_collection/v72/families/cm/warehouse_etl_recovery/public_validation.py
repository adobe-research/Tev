"""Executable public-rule validators for warehouse ETL recovery."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from training_data_collection.v72.contracts.public_task import PublicTaskSpecification
from training_data_collection.v72.qualification.public_contract import (
    PublicRuleViolation,
    PublicValidator,
    PublicValidatorContext,
)

from .definition import PUBLIC_RULE_IDS
from .verifier import WarehouseEvaluation, inspect_candidate


def build_validators(
    specification: PublicTaskSpecification,
    input_root: Path,
    output_root: Path,
) -> Mapping[str, PublicValidator]:
    """Cover every declared rule, relation, and precedence with semantic checks."""
    if (
        specification.environment.workspace_layout != "benchmark_native"
        or tuple(sorted(rule.rule_id for rule in specification.public_rules)) != PUBLIC_RULE_IDS
    ):
        raise ValueError("warehouse recovery public specification is not registered")
    evaluation: WarehouseEvaluation | None = None

    def require(*score_ids: str, scope: bool = False) -> PublicValidator:
        def validate(context: PublicValidatorContext) -> None:
            nonlocal evaluation
            if context.specification.specification_id != specification.specification_id:
                raise PublicRuleViolation("public validator context changed specification")
            if evaluation is None:
                evaluation = inspect_candidate(input_root, output_root)
            failures = [score_id for score_id in score_ids if evaluation.scores[score_id] < 1.0]
            if scope and not evaluation.output_scope_valid:
                failures.append("output_scope")
            if failures:
                raise PublicRuleViolation(f"warehouse semantic checks failed: {failures}")

        return validate

    validators = {
        "validator.dimensions": require("dimensions", "normalization"),
        "validator.ingestion": require("facts"),
        "validator.fact-load": require("aggregates", "relations"),
        "validator.output-scope": require("readable", scope=True),
        "validator.relational-quality": require("relations"),
        "validator.sidecars": require("quality_report", "summary"),
        "validator.dimension-normalization": require("dimensions", "normalization"),
        "validator.fact-reconciliation": require("facts"),
        "validator.fact-star-join": require("aggregates", "relations"),
        "validator.relational-validation": require("relations"),
        "validator.sidecar-truth": require("quality_report", "summary"),
        "validator.schema-precedence": require("dimensions", "normalization"),
    }
    return validators


__all__ = ["build_validators"]
