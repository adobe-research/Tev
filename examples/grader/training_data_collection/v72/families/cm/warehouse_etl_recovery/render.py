"""Mixed-format projections, artifact DAG, and controlled ETL corruptions."""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from training_data_collection.common.filesystem import resolve_under, tree_manifest, tree_sha256
from training_data_collection.common.identity import content_digest, content_id
from training_data_collection.v72.contracts.artifact import (
    ArtifactGraph,
    ArtifactNode,
    ArtifactRelation,
    RelationParameter,
)
from training_data_collection.v72.contracts.corruption import (
    CorruptionEvent,
    CorruptionOperatorDeclaration,
)
from training_data_collection.v72.contracts.world import LatentWorldManifest
from training_data_collection.v72.families.base import WorldStateView
from training_data_collection.v72.generation.corruption import MutableArtifactTree

from .world import (
    VALID_COUNTRY_CODES,
    VALID_SEGMENTS,
    VALID_SUPPLIERS,
    WORLD_ENTITY_ID,
    load_world,
)

RAW_TRANSACTIONS_PATH = "input/raw_transactions"
CUSTOMERS_PATH = "input/raw_customers/customers.json"
PRODUCTS_PATH = "input/raw_products/product_catalog.tsv"
SCHEMA_PATH = "input/target_schema_spec.json"
OUTPUT_CONTRACT_PATH = "input/output_contract.json"
RUNTIME_PATH = "input/runtime_env"
SOFTWARE_PATH = "software"
DATABASE_PATH = "output/warehouse.db"
QUALITY_REPORT_PATH = "output/data_quality_report.json"
SUMMARY_PATH = "output/warehouse_summary.json"

EXPECTED_TABLE_COLUMNS = {
    "dim_customers": {
        "country_code",
        "customer_id",
        "customer_key",
        "email",
        "name",
        "registration_date",
        "segment",
    },
    "dim_dates": {
        "date_key",
        "day",
        "day_of_week",
        "full_date",
        "is_weekend",
        "month",
        "quarter",
        "year",
    },
    "dim_products": {
        "base_price",
        "category",
        "cost_price",
        "is_active",
        "product_id",
        "product_key",
        "product_name",
        "subcategory",
        "supplier",
    },
    "fact_transactions": {
        "channel",
        "customer_key",
        "date_key",
        "discount_pct",
        "product_key",
        "quantity",
        "total_amount",
        "transaction_id",
        "unit_price",
    },
}

PUBLIC_ARTIFACT_NAMES = (
    "input.customers",
    "input.output-contract",
    "input.products",
    "input.runtime",
    "input.schema",
    "input.transactions",
    "output.data-quality-report",
    "output.warehouse",
    "output.warehouse-summary",
    "software.runtime-notes",
)
PUBLIC_RELATION_NAMES = (
    "relation.dimension-normalization",
    "relation.fact-reconciliation",
    "relation.fact-star-join",
    "relation.output-scope",
    "relation.relational-validation",
    "relation.sidecar-truth",
)

_COUNTRY_VARIANTS = {
    "CA": ("CA", "Canada", "can"),
    "DE": ("DE", "Deutschland", "Germany"),
    "GB": ("GB", "UK", "United Kingdom"),
    "US": ("U.S.", "USA", "United States", "us"),
}
_SEGMENT_VARIANTS = {
    "basic": (" BASIC ", "Basic", "basic"),
    "premium": ("PREMIUM", "Premium", "premium"),
    "standard": (" STANDARD", "Standard", "standard"),
}
_SUPPLIER_VARIANTS = {
    "Blue Mesa": ("BLUE MESA", "BlueMesa", "Blue_Mesa"),
    "Cedar Works": ("CEDAR WORKS", "Cedar-Works", "CedarWorks"),
    "Harbor Wholesale": ("HARBOR WHOLESALE", "HarborWholesale", "Harbor_Wholesale"),
    "Northstar Supply": ("NORTHSTAR SUPPLY", "NorthstarSupply", "Northstar_Supply"),
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _delimited_bytes(
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
    *,
    delimiter: str = ",",
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        delimiter=delimiter,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def schema_specification(world: Mapping[str, object]) -> dict[str, object]:
    return {
        "database_format": "sqlite",
        "logical_model_name": "retail_analytics",
        "date_dimension": {"start": world["date_start"], "end": world["date_end"]},
        "preserve_customer_ids": world["preserve_customer_ids"],
        "valid_customer_values": {
            "country_codes": list(VALID_COUNTRY_CODES),
            "segments": list(VALID_SEGMENTS),
        },
        "valid_supplier_names": list(VALID_SUPPLIERS),
        "tables": {
            table: sorted(columns) for table, columns in sorted(EXPECTED_TABLE_COLUMNS.items())
        },
        "notes": [
            "Use customer_id 0 for one synthetic Unknown Customer row.",
            "Use YYYYMMDD integer date keys.",
            "Label empty product categories Uncategorized.",
        ],
    }


def output_contract() -> dict[str, object]:
    return {
        "required_files": [
            "warehouse.db",
            "data_quality_report.json",
            "warehouse_summary.json",
        ],
        "data_quality_report": {
            "required_sections": {
                "extraction": [
                    "raw_transaction_files",
                    "raw_transaction_rows",
                    "raw_customer_records",
                    "raw_product_records",
                ],
                "transformations": [
                    "duplicates_removed",
                    "null_rows_dropped",
                    "null_customer_ids_mapped",
                    "timestamps_standardized",
                    "schema_drift_columns_filled",
                    "country_codes_standardized",
                    "supplier_names_standardized",
                    "boolean_fields_normalized",
                    "empty_categories_labeled",
                ],
                "load": [
                    "fact_transactions_rows",
                    "dim_customers_rows",
                    "dim_products_rows",
                    "dim_dates_rows",
                    "referential_integrity_violations_remaining",
                ],
                "quality_checks": [
                    "no_duplicate_transaction_ids",
                    "no_null_quantities",
                    "no_null_unit_prices",
                    "total_amount_formula_correct",
                    "all_foreign_keys_resolve",
                    "all_segments_valid",
                    "all_country_codes_iso2",
                ],
            }
        },
        "warehouse_summary": {
            "required_sections": {
                "fact_table": [
                    "row_count",
                    "unique_transactions",
                    "unique_customers",
                    "unique_products",
                    "total_revenue",
                    "avg_transaction_value",
                    "date_range",
                    "channel_distribution",
                ],
                "dim_customers": [
                    "row_count",
                    "segment_distribution",
                    "country_distribution",
                ],
                "dim_products": [
                    "row_count",
                    "category_distribution",
                    "active_products",
                ],
            }
        },
    }


def _world(view: WorldStateView) -> dict[str, object]:
    return load_world(view.entity_path(WORLD_ENTITY_ID))


def render_consistent_artifacts(view: WorldStateView, public_root: Path) -> None:
    world = _world(view)
    transactions = world["transactions"]
    customers = world["customers"]
    products = world["products"]
    if (
        not isinstance(transactions, list)
        or not isinstance(customers, list)
        or not isinstance(products, list)
    ):
        raise TypeError("warehouse world collections must be arrays")
    (public_root / RAW_TRANSACTIONS_PATH).mkdir(parents=True)
    (public_root / Path(CUSTOMERS_PATH).parent).mkdir(parents=True)
    (public_root / Path(PRODUCTS_PATH).parent).mkdir(parents=True)
    (public_root / RUNTIME_PATH).mkdir(parents=True)
    (public_root / SOFTWARE_PATH).mkdir(parents=True)
    columns = (
        "transaction_id",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
        "timestamp",
        "total_amount",
        "discount_pct",
        "channel",
    )
    for month in range(1, 7):
        rows = []
        for row in transactions:
            if int(row["month"]) != month:
                continue
            rows.append(
                {
                    **row,
                    "timestamp": datetime.fromisoformat(str(row["timestamp"])).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "total_amount": round(
                        int(row["quantity"])
                        * float(row["unit_price"])
                        * (1.0 - float(row["discount_pct"])),
                        2,
                    ),
                }
            )
        path = f"{RAW_TRANSACTIONS_PATH}/transactions_2025_{month:02d}.csv"
        resolve_under(public_root, path).write_bytes(_delimited_bytes(columns, rows))
    customer_rows = [
        {
            "id": row["customer_id"],
            "name": row["name"],
            "email": row["email"],
            "registration_date": row["registration_date"],
            "segment": row["segment"],
            "country": row["country_code"],
            "lifetime_value": 0.0,
        }
        for row in customers
    ]
    resolve_under(public_root, CUSTOMERS_PATH).write_bytes(_json_bytes(customer_rows))
    product_rows = [
        {
            **row,
            "weight_kg": 1.0,
            "created_at": "2024-01-01",
        }
        for row in products
    ]
    resolve_under(public_root, PRODUCTS_PATH).write_bytes(
        _delimited_bytes(
            (
                "product_id",
                "product_name",
                "category",
                "subcategory",
                "base_price",
                "cost_price",
                "supplier",
                "weight_kg",
                "is_active",
                "created_at",
            ),
            product_rows,
            delimiter="\t",
        )
    )
    resolve_under(public_root, SCHEMA_PATH).write_bytes(_json_bytes(schema_specification(world)))
    resolve_under(public_root, OUTPUT_CONTRACT_PATH).write_bytes(_json_bytes(output_contract()))
    resolve_under(public_root, f"{RUNTIME_PATH}/pyproject.toml").write_bytes(
        b'[project]\nname = "warehouse-etl-runtime"\nversion = "0.1.0"\n'
        b'requires-python = ">=3.12,<3.14"\ndependencies = []\n'
    )
    resolve_under(public_root, f"{SOFTWARE_PATH}/README.txt").write_bytes(
        b"Use the preinstalled python3 standard library and SQLite support.\n"
        b"Keep temporary files under output/_runtime if needed.\n"
    )


def validate_consistent_artifacts(view: WorldStateView, public_root: Path) -> bool:
    with Path(public_root).joinpath(CUSTOMERS_PATH).open(encoding="utf-8") as stream:
        customers = json.load(stream)
    with Path(public_root).joinpath(PRODUCTS_PATH).open(encoding="utf-8", newline="") as stream:
        products = list(csv.DictReader(stream, delimiter="\t"))
    world = _world(view)
    raw_rows = 0
    for path in sorted((public_root / RAW_TRANSACTIONS_PATH).glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            raw_rows += sum(1 for _ in csv.DictReader(stream))
    return bool(
        len(customers) == len(world["customers"])
        and len(products) == len(world["products"])
        and raw_rows == len(world["transactions"])
        and json.loads((public_root / SCHEMA_PATH).read_bytes()) == schema_specification(world)
        and json.loads((public_root / OUTPUT_CONTRACT_PATH).read_bytes()) == output_contract()
    )


def _artifact_nodes(world: LatentWorldManifest) -> dict[str, ArtifactNode]:
    definitions = {
        "customers": (CUSTOMERS_PATH, "file", "json", "controlled_corrupted_artifact", 30),
        "output_contract": (
            OUTPUT_CONTRACT_PATH,
            "file",
            "json",
            "schema_or_interface_contract",
            100,
        ),
        "products": (PRODUCTS_PATH, "file", "tsv", "controlled_corrupted_artifact", 30),
        "runtime": (RUNTIME_PATH, "directory", "toml", "configuration", 80),
        "schema": (SCHEMA_PATH, "file", "json", "schema_or_interface_contract", 100),
        "transactions": (
            RAW_TRANSACTIONS_PATH,
            "directory",
            "csv",
            "controlled_corrupted_artifact",
            30,
        ),
        "data_quality_report": (
            QUALITY_REPORT_PATH,
            "file",
            "json",
            "required_output",
            40,
        ),
        "warehouse": (DATABASE_PATH, "file", "sqlite", "required_output", 40),
        "warehouse_summary": (SUMMARY_PATH, "file", "json", "required_output", 40),
        "software": (SOFTWARE_PATH, "directory", "text", "executable_or_harness", 80),
    }
    nodes = {}
    for name, (path, path_kind, native_schema, role, rank) in definitions.items():
        nodes[name] = ArtifactNode.create(
            visibility="public",
            path=path,
            path_kind=path_kind,  # type: ignore[arg-type]
            artifact_type=native_schema,
            native_schema=f"warehouse-recovery-{native_schema}.v1",
            semantic_role=role,  # type: ignore[arg-type]
            world_projection_id=content_id(
                "warehouse-recovery-projection",
                {"world_id": world.world_id, "artifact": name, "renderer": 1},
            ),
            authority_rank=rank,
            mutable_by_policy=name in {"data_quality_report", "warehouse", "warehouse_summary"},
            source_record_ids=world.source_record_ids,
        )
    return nodes


def _relation(
    relation_type: str,
    sources: tuple[ArtifactNode, ...],
    targets: tuple[ArtifactNode, ...],
    *,
    rule: str,
    stage: str,
    reward: tuple[str, ...],
    validator: str,
    invariants: tuple[str, ...],
    parameters: tuple[RelationParameter, ...] = (),
) -> ArtifactRelation:
    return ArtifactRelation.create(
        relation_type=relation_type,  # type: ignore[arg-type]
        source_artifact_ids=tuple(item.artifact_id for item in sources),
        target_artifact_ids=tuple(item.artifact_id for item in targets),
        key_parameters=parameters,
        public_rule_ids=(rule,),
        invariant_ids=invariants,
        workflow_stage_ids=(stage,),
        reward_component_ids=reward,
        validator_implementation_id=validator,
        critical=True,
    )


def build_artifact_graph(world: LatentWorldManifest) -> ArtifactGraph:
    nodes = _artifact_nodes(world)
    invariants = tuple(item.invariant_id for item in world.invariant_declarations)
    relations = (
        _relation(
            "derived_from",
            (nodes["customers"], nodes["products"], nodes["schema"]),
            (nodes["warehouse"],),
            rule="rule.dimensions",
            stage="normalize-dimensions",
            reward=("reward.standardization",),
            validator="validator.dimension-normalization",
            invariants=invariants,
        ),
        _relation(
            "generated_from",
            (nodes["transactions"], nodes["schema"]),
            (nodes["warehouse"],),
            rule="rule.ingestion",
            stage="reconcile-facts",
            reward=("reward.row-counts",),
            validator="validator.fact-reconciliation",
            invariants=invariants,
            parameters=(RelationParameter(name="dedupe_winner", value="first"),),
        ),
        _relation(
            "joins_on",
            (nodes["transactions"], nodes["customers"], nodes["products"]),
            (nodes["warehouse"],),
            rule="rule.fact-load",
            stage="load-facts",
            reward=("reward.data-quality", "reward.revenue"),
            validator="validator.fact-star-join",
            invariants=invariants,
        ),
        _relation(
            "constrained_by",
            (nodes["output_contract"], nodes["runtime"], nodes["software"]),
            (nodes["warehouse"], nodes["data_quality_report"], nodes["warehouse_summary"]),
            rule="rule.output-scope",
            stage="package-sidecars",
            reward=("reward.schema",),
            validator="validator.output-scope",
            invariants=invariants,
        ),
        _relation(
            "validated_by",
            (nodes["schema"], nodes["transactions"]),
            (nodes["warehouse"],),
            rule="rule.relational-quality",
            stage="aggregate-validate",
            reward=("reward.data-quality", "reward.schema"),
            validator="validator.relational-validation",
            invariants=invariants,
        ),
        _relation(
            "summarizes",
            (
                nodes["transactions"],
                nodes["customers"],
                nodes["products"],
                nodes["output_contract"],
            ),
            (nodes["data_quality_report"], nodes["warehouse_summary"]),
            rule="rule.sidecars",
            stage="package-sidecars",
            reward=("reward.sidecars",),
            validator="validator.sidecar-truth",
            invariants=invariants,
        ),
    )
    inputs = tuple(
        nodes[name]
        for name in (
            "customers",
            "output_contract",
            "products",
            "runtime",
            "schema",
            "software",
            "transactions",
        )
    )
    return ArtifactGraph.create(
        world_id=world.world_id,
        nodes=tuple(nodes.values()),
        relations=relations,
        critical_public_artifact_ids=tuple(node.artifact_id for node in inputs),
        terminal_output_artifact_ids=(
            nodes["data_quality_report"].artifact_id,
            nodes["warehouse"].artifact_id,
            nodes["warehouse_summary"].artifact_id,
        ),
    )


def public_relation_id_map(graph: ArtifactGraph) -> Mapping[str, str]:
    if len(graph.relations) != len(PUBLIC_RELATION_NAMES):
        raise ValueError("warehouse recovery graph relation count is invalid")
    return dict(
        zip(PUBLIC_RELATION_NAMES, (item.relation_id for item in graph.relations), strict=True)
    )


def _read_csv_bytes(
    payload: bytes,
    *,
    delimiter: str = ",",
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter=delimiter)
    return tuple(reader.fieldnames or ()), list(reader)


@dataclass(frozen=True)
class PartitionSchemaDriftOperator:
    declaration: CorruptionOperatorDeclaration

    def apply(self, tree: MutableArtifactTree, perturbation_seed: int) -> None:
        del perturbation_seed
        for month in range(1, 7):
            path = f"{RAW_TRANSACTIONS_PATH}/transactions_2025_{month:02d}.csv"
            _, rows = _read_csv_bytes(tree.read_bytes(path))
            for row in rows:
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC
                )
                if month in (2, 4):
                    row["timestamp"] = timestamp.strftime("%m/%d/%Y %H:%M")
                elif month == 6:
                    row["timestamp"] = timestamp.strftime("%Y/%m/%d %H:%M:%S")
            columns = (
                "transaction_id",
                "customer_id",
                "product_id",
                "quantity",
                "unit_price",
                "timestamp",
                "total_amount",
            )
            if month >= 3:
                columns += ("discount_pct",)
            if month >= 5:
                columns += ("channel",)
            tree.write_bytes(path, _delimited_bytes(columns, rows))


@dataclass(frozen=True)
class FactDefectOperator:
    declaration: CorruptionOperatorDeclaration
    defects: Mapping[str, object]

    def apply(self, tree: MutableArtifactTree, perturbation_seed: int) -> None:
        del perturbation_seed
        rows_by_index: dict[int, dict[str, str]] = {}
        columns_by_month: dict[int, tuple[str, ...]] = {}
        month_rows: dict[int, list[dict[str, str]]] = {}
        for month in range(1, 7):
            path = f"{RAW_TRANSACTIONS_PATH}/transactions_2025_{month:02d}.csv"
            columns, rows = _read_csv_bytes(tree.read_bytes(path))
            columns_by_month[month] = columns
            month_rows[month] = rows
            for row in rows:
                rows_by_index[int(row["transaction_id"].rsplit("-", 1)[1])] = row
        for index in self.defects["null_measure_indices"]:  # type: ignore[index]
            row = rows_by_index[int(index)]
            row["quantity" if int(index) % 2 else "unit_price"] = ""
            row["total_amount"] = ""
        for index in self.defects["null_customer_indices"]:  # type: ignore[index]
            rows_by_index[int(index)]["customer_id"] = ""
        for index in self.defects["duplicate_indices"]:  # type: ignore[index]
            source = rows_by_index[int(index)]
            month_rows[6].append({column: source.get(column, "") for column in columns_by_month[6]})
        for month, rows in month_rows.items():
            path = f"{RAW_TRANSACTIONS_PATH}/transactions_2025_{month:02d}.csv"
            tree.write_bytes(path, _delimited_bytes(columns_by_month[month], rows))


@dataclass(frozen=True)
class DimensionDriftOperator:
    declaration: CorruptionOperatorDeclaration

    def apply(self, tree: MutableArtifactTree, perturbation_seed: int) -> None:
        rng_offset = perturbation_seed % 17
        customers = json.loads(tree.read_bytes(CUSTOMERS_PATH))
        for index, row in enumerate(customers):
            customer_id = int(row["id"])
            if customer_id >= 9001:
                if index % 3 == 0:
                    row["name"] = ""
                if index % 2:
                    row["email"] = ""
                row["registration_date"] = ""
                row["segment"] = ""
                row["country"] = None
                continue
            country = str(row["country"])
            segment = str(row["segment"])
            row["country"] = _COUNTRY_VARIANTS[country][
                (index + rng_offset) % len(_COUNTRY_VARIANTS[country])
            ]
            row["segment"] = _SEGMENT_VARIANTS[segment][
                (index + rng_offset) % len(_SEGMENT_VARIANTS[segment])
            ]
        tree.write_bytes(CUSTOMERS_PATH, _json_bytes(customers))
        columns, products = _read_csv_bytes(tree.read_bytes(PRODUCTS_PATH), delimiter="\t")
        for index, row in enumerate(products):
            supplier = row["supplier"]
            row["supplier"] = _SUPPLIER_VARIANTS[supplier][
                (index + rng_offset) % len(_SUPPLIER_VARIANTS[supplier])
            ]
            row["is_active"] = (
                ("0", "FALSE", "No", "n")[index % 4]
                if row["is_active"] == "0"
                else ("1", "TRUE", "yes", "Y")[index % 4]
            )
            if index % 11 == 0:
                row["category"] = ""
        tree.write_bytes(PRODUCTS_PATH, _delimited_bytes(columns, products, delimiter="\t"))


def corruption_operators(
    world: LatentWorldManifest,
    graph: ArtifactGraph,
) -> tuple[PartitionSchemaDriftOperator, FactDefectOperator, DimensionDriftOperator]:
    seed_declarations = [
        value.removeprefix("defect-plan-seed.")
        for value in world.renderer_declarations
        if value.startswith("defect-plan-seed.")
    ]
    transaction_declarations = [
        int(value.removeprefix("scale.transactions-per-month-"))
        for value in world.renderer_declarations
        if value.startswith("scale.transactions-per-month-")
    ]
    if len(seed_declarations) != 1 or len(transaction_declarations) != 1:
        raise ValueError("warehouse world lacks exact defect-plan authority")
    transaction_count = 6 * transaction_declarations[0]
    defect_rng = random.Random(int(seed_declarations[0], 16))
    defect_counts = {
        "duplicates": max(6, round(transaction_count * defect_rng.uniform(0.025, 0.035))),
        "null_customers": max(5, round(transaction_count * defect_rng.uniform(0.03, 0.04))),
        "null_measures": max(8, round(transaction_count * defect_rng.uniform(0.065, 0.08))),
    }
    null_measures = set(defect_rng.sample(range(transaction_count), defect_counts["null_measures"]))
    valid_measures = sorted(set(range(transaction_count)) - null_measures)
    null_customers = set(defect_rng.sample(valid_measures, defect_counts["null_customers"]))
    duplicates = sorted(
        defect_rng.sample(
            sorted(set(valid_measures) - null_customers),
            defect_counts["duplicates"],
        )
    )
    defects = {
        "duplicate_indices": duplicates,
        "null_customer_indices": sorted(null_customers),
        "null_measure_indices": sorted(null_measures),
    }
    artifacts = {node.path: node.artifact_id for node in graph.nodes}
    relation_ids = public_relation_id_map(graph)
    invariants = tuple(item.invariant_id for item in world.invariant_declarations)

    def declaration(
        operator_id: str,
        affected_paths: tuple[str, ...],
        relations: tuple[str, ...],
        stages: tuple[str, ...],
        effects: tuple[str, ...],
    ) -> CorruptionOperatorDeclaration:
        return CorruptionOperatorDeclaration.identified(
            operator_id=operator_id,
            implementation_digest=content_digest({"operator": operator_id, "version": 1}),
            affected_artifact_ids=tuple(sorted(artifacts[path] for path in affected_paths)),
            affected_relation_ids=tuple(sorted(relation_ids[name] for name in relations)),
            affected_invariant_ids=invariants,
            affected_stage_ids=tuple(sorted(stages)),
            expected_metamorphic_effects=tuple(sorted(effects)),
        )

    schema = declaration(
        "warehouse.partition-schema-drift.v1",
        (RAW_TRANSACTIONS_PATH,),
        ("relation.fact-reconciliation", "relation.fact-star-join"),
        ("profile-sources", "reconcile-facts"),
        (
            "early partitions omit channel and discount_pct",
            "timestamp representations vary while preserving instants",
        ),
    )
    facts = declaration(
        "warehouse.duplicate-and-null-facts.v1",
        (RAW_TRANSACTIONS_PATH,),
        ("relation.fact-reconciliation", "relation.fact-star-join"),
        ("load-facts", "reconcile-facts"),
        (
            "duplicate transaction identities require deterministic first-row retention",
            "missing required measures require rejection",
            "missing customer identities map to Unknown Customer",
        ),
    )
    dimensions = declaration(
        "warehouse.dimension-category-drift.v1",
        (CUSTOMERS_PATH, PRODUCTS_PATH),
        ("relation.dimension-normalization", "relation.sidecar-truth"),
        ("normalize-dimensions", "package-sidecars"),
        (
            "customer countries and segments require canonicalization",
            "product suppliers, booleans, and blank categories require canonicalization",
        ),
    )
    return (
        PartitionSchemaDriftOperator(schema),
        FactDefectOperator(facts, defects),
        DimensionDriftOperator(dimensions),
    )


def _digests(root: Path) -> dict[str, str]:
    return {str(record["path"]): str(record["sha256"]) for record in tree_manifest(root)}


def apply_registered_corruptions(
    world: LatentWorldManifest,
    graph: ArtifactGraph,
    public_root: Path,
) -> tuple[ArtifactGraph, tuple[CorruptionEvent, ...]]:
    perturbation_seed = int(
        content_digest({"world_id": world.world_id, "namespace": "perturbation"})[:32], 16
    )
    baseline_ids = tuple(sorted(node.artifact_id for node in graph.nodes))
    tree = MutableArtifactTree(public_root)
    events = []
    for operator in corruption_operators(world, graph):
        before = _digests(public_root)
        before_tree = tree_sha256(public_root)
        operator.apply(tree, perturbation_seed)
        after = _digests(public_root)
        after_tree = tree_sha256(public_root)
        changed_paths = tuple(
            sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        )
        if not changed_paths:
            raise ValueError("warehouse corruption operator had no realized effect")
        events.append(
            CorruptionEvent.identified(
                world_id=world.world_id,
                operator_id=operator.declaration.operator_id,
                operator_declaration_id=operator.declaration.declaration_id,
                baseline_artifact_ids=baseline_ids,
                affected_artifact_ids=operator.declaration.affected_artifact_ids,
                affected_relation_ids=operator.declaration.affected_relation_ids,
                affected_invariant_ids=operator.declaration.affected_invariant_ids,
                affected_stage_ids=operator.declaration.affected_stage_ids,
                expected_metamorphic_effects=operator.declaration.expected_metamorphic_effects,
                baseline_tree_sha256=before_tree,
                corrupted_tree_sha256=after_tree,
                changed_paths=changed_paths,
            )
        )
    return graph, tuple(events)


__all__ = [
    "CUSTOMERS_PATH",
    "DATABASE_PATH",
    "EXPECTED_TABLE_COLUMNS",
    "OUTPUT_CONTRACT_PATH",
    "PRODUCTS_PATH",
    "PUBLIC_ARTIFACT_NAMES",
    "PUBLIC_RELATION_NAMES",
    "QUALITY_REPORT_PATH",
    "RAW_TRANSACTIONS_PATH",
    "RUNTIME_PATH",
    "SCHEMA_PATH",
    "SOFTWARE_PATH",
    "SUMMARY_PATH",
    "apply_registered_corruptions",
    "build_artifact_graph",
    "output_contract",
    "public_relation_id_map",
    "render_consistent_artifacts",
    "schema_specification",
    "validate_consistent_artifacts",
]
