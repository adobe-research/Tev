"""Continuous terminal reward authority and causal output mutations."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from training_data_collection.common.filesystem import tree_sha256
from training_data_collection.common.identity import content_digest, content_id
from training_data_collection.v72.contracts.reward import RewardComponent, RewardContract
from training_data_collection.v72.qualification.reward import (
    RewardMutationExecution,
    RewardMutationRegistry,
)

from .verifier import verify_warehouse_etl

_AUTHORITIES = (
    (
        "reward.data-quality",
        "rule.relational-quality",
        "aggregate-validate",
        "mutation.break-foreign-key",
        ("reward.revenue", "reward.sidecars"),
    ),
    (
        "reward.revenue",
        "rule.fact-load",
        "aggregate-validate",
        "mutation.change-fact-amount",
        ("reward.data-quality", "reward.sidecars"),
    ),
    (
        "reward.row-counts",
        "rule.ingestion",
        "reconcile-facts",
        "mutation.delete-fact-row",
        ("reward.revenue", "reward.sidecars"),
    ),
    (
        "reward.schema",
        "rule.output-scope",
        "profile-sources",
        "mutation.drop-schema-column",
        (),
    ),
    (
        "reward.sidecars",
        "rule.sidecars",
        "package-sidecars",
        "mutation.stale-summary",
        (),
    ),
    (
        "reward.standardization",
        "rule.dimensions",
        "normalize-dimensions",
        "mutation.denormalize-country",
        ("reward.sidecars",),
    ),
)


def build_reward_contract() -> RewardContract:
    """Build six independently recomputed, equally weighted reward components."""
    return RewardContract.identified(
        components=tuple(
            RewardComponent(
                component_id=component_id,
                description=f"Continuously score {component_id.removeprefix('reward.')}.",
                weight=1 / 6,
                public_rule_ids=(rule_id,),
                workflow_stage_ids=(stage_id,),
                checker_implementation_id=content_id(
                    "v72-warehouse-recovery-reward-checker",
                    {"component_id": component_id, "version": 1},
                ),
                causal_mutation_ids=(mutation_id,),
                allowed_collateral_component_ids=collateral,
            )
            for component_id, rule_id, stage_id, mutation_id, collateral in _AUTHORITIES
        ),
        hard_gate_public_rule_ids=("rule.output-scope",),
    )


def _mutate(component_id: str, output_root: Path) -> None:
    if component_id == "reward.sidecars":
        path = output_root / "warehouse_summary.json"
        value = json.loads(path.read_bytes())
        value["fact_table"]["row_count"] += 1
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return
    database = output_root / "warehouse.db"
    with sqlite3.connect(database) as connection:
        if component_id == "reward.data-quality":
            connection.execute("pragma foreign_keys=off")
            connection.execute(
                "update fact_transactions set customer_key=999999 where transaction_id="
                "(select min(transaction_id) from fact_transactions)"
            )
        elif component_id == "reward.revenue":
            connection.execute(
                "update fact_transactions set total_amount=total_amount+1000000 where "
                "transaction_id=(select min(transaction_id) from fact_transactions)"
            )
        elif component_id == "reward.row-counts":
            connection.execute(
                "delete from fact_transactions where transaction_id="
                "(select min(transaction_id) from fact_transactions)"
            )
        elif component_id == "reward.schema":
            connection.execute("alter table dim_products drop column subcategory")
        elif component_id == "reward.standardization":
            connection.execute(
                "update dim_customers set country_code='United States' where customer_key="
                "(select min(customer_key) from dim_customers)"
            )
        else:
            raise ValueError(f"unknown warehouse reward component: {component_id}")


def build_reward_mutation_registry(
    *,
    task_id: str,
    public_root: Path,
    baseline_output_root: Path,
    mutation_work_root: Path,
) -> tuple[RewardContract, RewardMutationRegistry]:
    """Bind each reward component to an executed, local causal mutation."""
    contract = build_reward_contract()
    baseline = verify_warehouse_etl(
        task_id=task_id,
        reward_contract=contract,
        public_input_root=public_root,
        sealed_output_root=baseline_output_root,
    )
    if baseline.status != "policy_result" or baseline.total != 1.0:
        raise ValueError("reward mutation qualification requires a perfect baseline")
    mutation_work_root.mkdir(parents=True, exist_ok=False)
    baseline_digest = tree_sha256(baseline_output_root)
    registry = RewardMutationRegistry()
    for component_id, _, _, mutation_id, _ in _AUTHORITIES:

        def execute(
            requested: RewardComponent,
            *,
            expected_component_id: str = component_id,
        ) -> RewardMutationExecution:
            if requested.component_id != expected_component_id:
                raise ValueError("mutation requested for a different reward component")
            destination = mutation_work_root / expected_component_id.removeprefix("reward.")
            shutil.copytree(baseline_output_root, destination)
            _mutate(expected_component_id, destination)
            mutated_digest = tree_sha256(destination)
            mutated = verify_warehouse_etl(
                task_id=task_id,
                reward_contract=contract,
                public_input_root=public_root,
                sealed_output_root=destination,
            )
            return RewardMutationExecution(
                baseline_result=baseline,
                mutated_result=mutated,
                baseline_output_sha256=baseline_digest,
                mutated_output_sha256=mutated_digest,
                execution_evidence_ids=(
                    content_id(
                        "v72-warehouse-recovery-mutation-run",
                        {
                            "component_id": expected_component_id,
                            "baseline": baseline_digest,
                            "mutated": mutated_digest,
                        },
                    ),
                ),
            )

        registry.register(
            mutation_id,
            content_digest(
                {"implementation": "warehouse-recovery-mutation", "component": component_id}
            ),
            execute,
        )
    return contract, registry


__all__ = ["build_reward_contract", "build_reward_mutation_registry"]
