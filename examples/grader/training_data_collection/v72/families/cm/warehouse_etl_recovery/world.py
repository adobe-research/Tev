"""Fresh latent retail worlds for the benchmark-shaped v7.2 ETL family."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from training_data_collection.common.identity import canonical_json, content_digest, content_id
from training_data_collection.v72.contracts.identity import SeedNamespace
from training_data_collection.v72.contracts.world import (
    WorldEntityState,
    WorldInvariant,
    WorldRecipe,
)
from training_data_collection.v72.generation.world import (
    WorldBuildContext,
    WorldBuildResult,
    WorldInvariantValidator,
    materialize_world,
)

FAMILY_REVISION_ID = "cm.warehouse-etl.v2"
WORLD_ENTITY_ID = "entity.retail-warehouse"
WORLD_STATE_PATH = "state/retail_warehouse.json"

VALID_SEGMENTS = ("basic", "premium", "standard", "unknown")
VALID_COUNTRY_CODES = ("CA", "DE", "GB", "US", "XX")
VALID_SUPPLIERS = (
    "Blue Mesa",
    "Cedar Works",
    "Harbor Wholesale",
    "Northstar Supply",
)
DATE_START = date(2025, 1, 1)
DATE_END = date(2025, 6, 30)


@dataclass(frozen=True)
class WarehouseRecoveryScale:
    scale_id: str
    customer_count: int
    provisional_customer_count: int
    product_count: int
    transactions_per_month: int


UNIT_SCALE = WarehouseRecoveryScale("unit", 30, 4, 12, 48)
RECOVERY_SCALE = WarehouseRecoveryScale("recovery", 480, 15, 52, 960)
SCALES: Mapping[str, WarehouseRecoveryScale] = {
    UNIT_SCALE.scale_id: UNIT_SCALE,
    RECOVERY_SCALE.scale_id: RECOVERY_SCALE,
}

WORLD_INVARIANT_VALIDATOR_ID = content_id(
    "warehouse-recovery-world-validator",
    {"implementation": "retail-star-schema-authority", "version": 1},
)


def _resolve_scale(recipe: WorldRecipe) -> WarehouseRecoveryScale:
    try:
        default = SCALES[recipe.scale_id]
    except KeyError as error:
        raise ValueError(f"unknown warehouse recovery scale: {recipe.scale_id}") from error
    parameters = recipe.parameter_dict()
    allowed = {
        "customer_count",
        "provisional_customer_count",
        "product_count",
        "transactions_per_month",
    }
    if set(parameters) - allowed:
        raise ValueError("warehouse recovery recipe has unknown scale parameters")
    values = {name: parameters.get(name, getattr(default, name)) for name in sorted(allowed)}
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError("warehouse recovery scale values must be positive integers")
    if values["customer_count"] < 8 or values["product_count"] < 5:
        raise ValueError("warehouse recovery scale lacks categorical coverage")
    if values["transactions_per_month"] < 20:
        raise ValueError("warehouse recovery scale lacks meaningful partition depth")
    return WarehouseRecoveryScale(scale_id=recipe.scale_id, **values)


def _build_world_result(
    recipe: WorldRecipe,
    rng: random.Random,
    governed_sources: tuple[tuple[str, bytes], ...],
) -> WorldBuildResult:
    scale = _resolve_scale(recipe)
    customer_count = scale.customer_count + (
        rng.randrange(16) if scale.scale_id == "recovery" else 0
    )
    product_count = scale.product_count + (rng.randrange(4) if scale.scale_id == "recovery" else 0)
    transactions_per_month = scale.transactions_per_month + (
        rng.randrange(81) if scale.scale_id == "recovery" else 0
    )
    countries = VALID_COUNTRY_CODES[:-1]
    segments = VALID_SEGMENTS[:-1]
    customers: list[dict[str, object]] = []
    for index in range(customer_count):
        customer_id = 1001 + index
        customers.append(
            {
                "customer_id": customer_id,
                "name": f"Buyer {customer_id}",
                "email": f"buyer{customer_id}@shop.example",
                "registration_date": (
                    date(2021, 1, 1) + timedelta(days=rng.randrange(1400))
                ).isoformat(),
                "segment": segments[rng.randrange(len(segments))],
                "country_code": countries[rng.randrange(len(countries))],
            }
        )
    provisional_ids = tuple(range(9001, 9001 + scale.provisional_customer_count))
    customers.extend(
        {
            "customer_id": customer_id,
            "name": f"Provisional {customer_id}",
            "email": f"pending{customer_id}@shop.example",
            "registration_date": "",
            "segment": "unknown",
            "country_code": "XX",
        }
        for customer_id in provisional_ids
    )

    categories = (
        ("Books", "Reference"),
        ("Clothing", "Outerwear"),
        ("Electronics", "Accessories"),
        ("Home", "Kitchen"),
        ("Sports", "Outdoor"),
    )
    products: list[dict[str, object]] = []
    for index in range(product_count):
        product_id = 3001 + index
        category, subcategory = categories[index % len(categories)]
        base_price = round(8.0 + index * 4.75 + rng.uniform(0.0, 20.0), 2)
        products.append(
            {
                "product_id": product_id,
                "product_name": f"Catalog Item {product_id}",
                "category": category,
                "subcategory": subcategory,
                "base_price": base_price,
                "cost_price": round(base_price * rng.uniform(0.45, 0.78), 2),
                "supplier": VALID_SUPPLIERS[index % len(VALID_SUPPLIERS)],
                "is_active": 0 if index % 7 == 0 else 1,
            }
        )

    transaction_count = 6 * transactions_per_month
    defect_plan_seed = rng.getrandbits(128)
    defect_rng = random.Random(defect_plan_seed)
    defect_counts = {
        "duplicates": max(6, round(transaction_count * defect_rng.uniform(0.025, 0.035))),
        "null_customers": max(5, round(transaction_count * defect_rng.uniform(0.03, 0.04))),
        "null_measures": max(8, round(transaction_count * defect_rng.uniform(0.065, 0.08))),
    }
    null_measure_indices = set(
        defect_rng.sample(range(transaction_count), defect_counts["null_measures"])
    )
    valid_measure_indices = sorted(set(range(transaction_count)) - null_measure_indices)
    null_customer_indices = set(
        defect_rng.sample(valid_measure_indices, defect_counts["null_customers"])
    )
    duplicate_candidates = sorted(set(valid_measure_indices) - null_customer_indices)
    duplicate_indices = tuple(
        sorted(defect_rng.sample(duplicate_candidates, defect_counts["duplicates"]))
    )
    transactions: list[dict[str, object]] = []
    customer_ids = [int(row["customer_id"]) for row in customers]
    for index in range(transaction_count):
        month = index // transactions_per_month + 1
        timestamp = datetime(
            2025,
            month,
            index % 28 + 1,
            index % 24,
            (index * 7) % 60,
            tzinfo=UTC,
        )
        product = products[rng.randrange(len(products))]
        quantity = 1 + rng.randrange(8)
        unit_price = round(float(product["base_price"]) * rng.uniform(0.92, 1.08), 2)
        discount = rng.choice((0.0, 0.0, 0.03, 0.05, 0.1))
        transactions.append(
            {
                "row_index": index,
                "month": month,
                "transaction_id": f"SALE-{month:02d}-{index:06d}",
                "customer_id": customer_ids[rng.randrange(len(customer_ids))],
                "product_id": int(product["product_id"]),
                "quantity": quantity,
                "unit_price": unit_price,
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "discount_pct": discount,
                "channel": rng.choice(("in_store", "mobile", "online", "partner")),
            }
        )
    world = {
        "schema_version": "warehouse-recovery-world.v2",
        "customers": customers,
        "products": products,
        "transactions": transactions,
        "defects": {
            **defect_counts,
            "duplicate_indices": list(duplicate_indices),
            "null_customer_indices": sorted(null_customer_indices),
            "null_measure_indices": sorted(null_measure_indices),
            "schema_drift_months": [1, 2, 3, 4],
        },
        "preserve_customer_ids": list(provisional_ids),
        "date_start": DATE_START.isoformat(),
        "date_end": DATE_END.isoformat(),
        "scale": {
            "customer_count": customer_count,
            "product_count": product_count,
            "transactions_per_month": transactions_per_month,
        },
    }
    invariant = WorldInvariant.identified(
        domain_description=(
            "Canonical retail dimensions and facts have unique identities, valid joins, "
            "finite measures, and a seed-derived nonempty defect plan."
        ),
        validator_implementation_id=WORLD_INVARIANT_VALIDATOR_ID,
        affected_entity_ids=(WORLD_ENTITY_ID,),
        related_public_rule_ids=tuple(
            sorted(
                (
                    "rule.dimensions",
                    "rule.fact-load",
                    "rule.ingestion",
                    "rule.output-scope",
                    "rule.relational-quality",
                    "rule.sidecars",
                )
            )
        ),
        related_workflow_stage_ids=tuple(
            sorted(
                (
                    "aggregate-validate",
                    "load-facts",
                    "normalize-dimensions",
                    "package-sidecars",
                    "profile-sources",
                    "reconcile-facts",
                )
            )
        ),
        affected_reward_component_ids=tuple(
            sorted(
                (
                    "reward.data-quality",
                    "reward.revenue",
                    "reward.row-counts",
                    "reward.schema",
                    "reward.sidecars",
                    "reward.standardization",
                )
            )
        ),
    )
    source_binding = content_digest(
        [
            {"source_record_id": source_id, "content_sha256": content_digest(value.hex())}
            for source_id, value in governed_sources
        ]
    )
    return WorldBuildResult(
        files={WORLD_STATE_PATH: canonical_json(world)},
        entity_index=(WorldEntityState(entity_id=WORLD_ENTITY_ID, state_path=WORLD_STATE_PATH),),
        invariants=(invariant,),
        renderer_declarations=tuple(
            sorted(
                (
                    "renderer.warehouse-etl-recovery.mixed-format.v1",
                    f"defect-plan-seed.{defect_plan_seed:032x}",
                    f"scale.customers-{customer_count}",
                    f"scale.products-{product_count}",
                    f"scale.transactions-per-month-{transactions_per_month}",
                    f"source-binding.{source_binding}",
                )
            )
        ),
    )


def _validate_world(state_root: Path, invariant: WorldInvariant) -> bool:
    if invariant.validator_implementation_id != WORLD_INVARIANT_VALIDATOR_ID:
        return False
    try:
        world = json.loads((state_root / WORLD_STATE_PATH).read_bytes())
        customers = world["customers"]
        products = world["products"]
        transactions = world["transactions"]
        defects = world["defects"]
        customer_ids = [int(row["customer_id"]) for row in customers]
        product_ids = [int(row["product_id"]) for row in products]
        transaction_ids = [str(row["transaction_id"]) for row in transactions]
        index_set = set(range(len(transactions)))
        defect_sets = {
            name: set(defects[f"{name}_indices"])
            for name in ("duplicate", "null_customer", "null_measure")
        }
        return bool(
            world["schema_version"] == "warehouse-recovery-world.v2"
            and len(customer_ids) == len(set(customer_ids))
            and len(product_ids) == len(set(product_ids))
            and len(transaction_ids) == len(set(transaction_ids))
            and all(int(row["customer_id"]) in set(customer_ids) for row in transactions)
            and all(int(row["product_id"]) in set(product_ids) for row in transactions)
            and all(int(row["quantity"]) > 0 for row in transactions)
            and all(float(row["unit_price"]) > 0 for row in transactions)
            and all(values and values.issubset(index_set) for values in defect_sets.values())
            and not defect_sets["null_measure"] & defect_sets["duplicate"]
            and not defect_sets["null_measure"] & defect_sets["null_customer"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


class WarehouseEtlRecoveryWorldMaterializer:
    implementation_digest = content_digest(
        {"implementation": "warehouse-etl-recovery-world", "version": 1}
    )

    def build(self, recipe: WorldRecipe, context: WorldBuildContext) -> WorldBuildResult:
        rng = context.random(SeedNamespace.PRIVATE_WORLD)
        sources = tuple(
            (source_id, context.read_source(source_id)) for source_id in recipe.source_record_ids
        )
        return _build_world_result(recipe, rng, sources)


def world_invariant_validators() -> Mapping[str, WorldInvariantValidator]:
    return {WORLD_INVARIANT_VALIDATOR_ID: _validate_world}


def load_world(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict or value.get("schema_version") != "warehouse-recovery-world.v2":
        raise ValueError("warehouse recovery world state is invalid")
    return value


def run_materializer_process(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {"recipe", "destination", "governed_sources"}:
        raise ValueError("materializer process payload is invalid")
    if (
        type(payload["recipe"]) is not dict
        or type(payload["destination"]) is not str
        or type(payload["governed_sources"]) is not dict
    ):
        raise ValueError("materializer process payload types are invalid")
    recipe = WorldRecipe.model_validate_json(canonical_json(payload["recipe"]))
    sources = {
        str(source_id): bytes.fromhex(str(encoded))
        for source_id, encoded in payload["governed_sources"].items()
    }
    result = materialize_world(
        recipe=recipe,
        materializer=WarehouseEtlRecoveryWorldMaterializer(),
        destination=Path(payload["destination"]),
        invariant_validators=world_invariant_validators(),
        governed_sources=sources,
    )
    return result.manifest.model_dump(mode="json")


__all__ = [
    "DATE_END",
    "DATE_START",
    "FAMILY_REVISION_ID",
    "RECOVERY_SCALE",
    "SCALES",
    "UNIT_SCALE",
    "VALID_COUNTRY_CODES",
    "VALID_SEGMENTS",
    "VALID_SUPPLIERS",
    "WORLD_ENTITY_ID",
    "WORLD_STATE_PATH",
    "WarehouseEtlRecoveryWorldMaterializer",
    "load_world",
    "run_materializer_process",
    "world_invariant_validators",
]
