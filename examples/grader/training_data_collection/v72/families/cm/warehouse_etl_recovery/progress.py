"""Warehouse-specific semantic progress DAG and checker registry."""

from __future__ import annotations

from collections.abc import Mapping

from training_data_collection.common.identity import content_id
from training_data_collection.v72.contracts.artifact import ArtifactGraph
from training_data_collection.v72.contracts.progress import ProgressContract, ProgressNode
from training_data_collection.v72.contracts.workflow import WorkflowGraph
from training_data_collection.v72.runtime.progress import CheckerMeasurement, ProgressChecker

from .reward import build_reward_contract
from .verifier import inspect_candidate
from .world import FAMILY_REVISION_ID

_NODES = {
    "aggregates": {
        "description": "Revenue and aggregate metrics match the reconciled facts.",
        "weight": 0.10,
        "prerequisites": ("facts", "relations"),
        "selectors": ("output/warehouse.db",),
        "rule": "rule.fact-load",
        "mutation": "mutation.change-fact-amount",
        "cost": "moderate",
    },
    "dimensions": {
        "description": "Customer, product, and calendar dimensions are complete.",
        "weight": 0.15,
        "prerequisites": ("schema",),
        "selectors": ("output/warehouse.db",),
        "rule": "rule.dimensions",
        "mutation": "mutation.delete-dimension-row",
        "cost": "moderate",
    },
    "exact": {
        "description": "All six benchmark-aligned terminal criteria pass together.",
        "weight": 0.0,
        "prerequisites": ("quality_report", "schema", "summary"),
        "selectors": ("output",),
        "rule": "rule.output-scope",
        "mutation": "mutation.add-undeclared-output",
        "cost": "expensive",
    },
    "facts": {
        "description": "Accepted fact row counts match public-source reconstruction.",
        "weight": 0.18,
        "prerequisites": ("schema",),
        "selectors": ("output/warehouse.db",),
        "rule": "rule.ingestion",
        "mutation": "mutation.delete-fact-row",
        "cost": "moderate",
    },
    "normalization": {
        "description": "Categorical and boolean dimension values are canonical.",
        "weight": 0.10,
        "prerequisites": ("dimensions",),
        "selectors": ("output/warehouse.db",),
        "rule": "rule.dimensions",
        "mutation": "mutation.denormalize-country",
        "cost": "moderate",
    },
    "quality_report": {
        "description": "The data quality report truthfully describes source handling and load.",
        "weight": 0.10,
        "prerequisites": ("aggregates", "relations"),
        "selectors": ("output/data_quality_report.json", "output/warehouse.db"),
        "rule": "rule.sidecars",
        "mutation": "mutation.stale-quality-report",
        "cost": "moderate",
    },
    "readable": {
        "description": "Required SQLite and JSON artifacts are independently readable.",
        "weight": 0.03,
        "prerequisites": (),
        "selectors": ("output",),
        "rule": "rule.output-scope",
        "mutation": "mutation.truncate-database",
        "cost": "cheap",
    },
    "relations": {
        "description": "Star-schema foreign keys and Unknown Customer mappings resolve.",
        "weight": 0.10,
        "prerequisites": ("dimensions", "facts"),
        "selectors": ("output/warehouse.db",),
        "rule": "rule.relational-quality",
        "mutation": "mutation.break-foreign-key",
        "cost": "moderate",
    },
    "schema": {
        "description": "All four required tables expose the exact declared columns.",
        "weight": 0.12,
        "prerequisites": ("readable",),
        "selectors": ("output/warehouse.db",),
        "rule": "rule.output-scope",
        "mutation": "mutation.drop-schema-column",
        "cost": "cheap",
    },
    "summary": {
        "description": "The warehouse summary exactly reflects candidate warehouse aggregates.",
        "weight": 0.12,
        "prerequisites": ("aggregates", "normalization"),
        "selectors": ("output/warehouse.db", "output/warehouse_summary.json"),
        "rule": "rule.sidecars",
        "mutation": "mutation.stale-summary",
        "cost": "moderate",
    },
}


def _artifact_ids(graph: ArtifactGraph) -> Mapping[str, str]:
    return {node.path: node.artifact_id for node in graph.nodes}


def build_progress_contract(
    graph: ArtifactGraph,
    workflow: WorkflowGraph,
) -> ProgressContract:
    """Bind the ten-node ETL progress DAG to workflow and final reward authority."""
    artifacts = _artifact_ids(graph)
    database = artifacts["output/warehouse.db"]
    quality = artifacts["output/data_quality_report.json"]
    summary = artifacts["output/warehouse_summary.json"]
    observed = {
        "quality_report": (database, quality),
        "summary": (database, summary),
        "exact": (database, quality, summary),
    }
    nodes = []
    for node_id, authority in sorted(_NODES.items()):
        prerequisites = authority["prerequisites"]
        nodes.append(
            ProgressNode.identified(
                node_id=node_id,
                description=authority["description"],
                observed_artifact_ids=tuple(sorted(observed.get(node_id, (database,)))),
                prerequisite_node_ids=prerequisites,
                gate_policy=(
                    "independent"
                    if not prerequisites
                    else "hard"
                    if node_id == "exact"
                    else "smooth_cap"
                ),
                prerequisite_threshold=1.0,
                checker_implementation_id=content_id(
                    "v72-warehouse-recovery-progress-checker",
                    {"node_id": node_id, "version": 1},
                ),
                continuous_score_definition=(
                    "Independent public-source recomputation mapped to a finite [0,1] fraction."
                ),
                weight=authority["weight"],
                causal_mutation_ids=(authority["mutation"],),
                allowed_collateral_node_ids=(),
                input_selectors=tuple(sorted(authority["selectors"])),
                online_evaluation_cost=authority["cost"],
                public_rule_ids=(authority["rule"],),
            )
        )
    return ProgressContract.identified(
        family_revision_id=FAMILY_REVISION_ID,
        workflow_id=workflow.workflow_id,
        final_reward_contract_id=build_reward_contract().contract_id,
        nodes=tuple(nodes),
        shaping_scale=0.25,
    )


def build_progress_checker_registry(
    contract: ProgressContract,
) -> dict[str, ProgressChecker]:
    """Return exact checker coverage for the content-bound progress contract."""
    registry: dict[str, ProgressChecker] = {}
    for node in contract.nodes:

        def check(public_root, output_root, private_root, *, node_id: str = node.node_id):
            del private_root
            evaluation = inspect_candidate(public_root, output_root)
            return CheckerMeasurement(
                raw_value=evaluation.scores[node_id],
                evidence_record_ids=(evaluation.evidence_record_id,),
            )

        registry[node.checker_implementation_id] = check
    return registry


__all__ = ["build_progress_checker_registry", "build_progress_contract"]
