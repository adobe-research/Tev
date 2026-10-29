"""Independent semantic verifier and continuous progress measurements."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from training_data_collection.common.filesystem import iter_regular_files
from training_data_collection.common.identity import canonical_json, content_id
from training_data_collection.v72.contracts.reward import RewardContract, RewardResult

_COMPONENTS = {
    "reward.data-quality",
    "reward.revenue",
    "reward.row-counts",
    "reward.schema",
    "reward.sidecars",
    "reward.standardization",
}
_COUNTRIES = {"CA", "DE", "GB", "US", "XX"}
_SEGMENTS = {"basic", "premium", "standard", "unknown"}
_SUPPLIERS = {"Blue Mesa", "Cedar Works", "Harbor Wholesale", "Northstar Supply"}
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


@dataclass(frozen=True)
class PublicExpectations:
    raw_files: int
    raw_rows: int
    raw_customers: int
    raw_products: int
    duplicate_rows: int
    missing_measure_rows: int
    missing_customer_rows: int
    fact_rows: int
    customer_rows: int
    product_rows: int
    date_rows: int
    revenue: float
    preserved_customer_ids: tuple[int, ...]


@dataclass(frozen=True)
class WarehouseEvaluation:
    scores: dict[str, float]
    terminal_criteria: dict[str, bool]
    output_scope_valid: bool
    evidence_record_id: str
    reason: str


def _closeness(actual: float, expected: float, tolerance: float) -> float:
    difference = abs(actual - expected)
    allowance = max(abs(expected) * tolerance, 1e-12)
    return max(0.0, 1.0 - difference / allowance)


def _public_expectations(public_root: Path) -> PublicExpectations:
    root = public_root.resolve(strict=True)
    if (root / "input").is_dir():
        root = (root / "input").resolve(strict=True)
    schema = json.loads((root / "target_schema_spec.json").read_bytes())
    customers = json.loads((root / "raw_customers/customers.json").read_bytes())
    if type(customers) is not list:
        raise ValueError("customer source is not an array")
    with (root / "raw_products/product_catalog.tsv").open(encoding="utf-8", newline="") as stream:
        products = list(csv.DictReader(stream, delimiter="\t"))
    paths = tuple(sorted((root / "raw_transactions").glob("transactions_*.csv")))
    seen: set[str] = set()
    raw_rows = duplicate_rows = missing_measures = missing_customers = accepted = 0
    revenue = 0.0
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                raw_rows += 1
                transaction_id = str(row.get("transaction_id") or "").strip()
                if transaction_id in seen:
                    duplicate_rows += 1
                    continue
                seen.add(transaction_id)
                quantity = str(row.get("quantity") or "").strip()
                price = str(row.get("unit_price") or "").strip()
                if not quantity or not price:
                    missing_measures += 1
                    continue
                if not str(row.get("customer_id") or "").strip():
                    missing_customers += 1
                accepted += 1
                revenue += round(
                    int(float(quantity))
                    * float(price)
                    * (1.0 - float(row.get("discount_pct") or 0.0)),
                    2,
                )
    start = date.fromisoformat(schema["date_dimension"]["start"])
    end = date.fromisoformat(schema["date_dimension"]["end"])
    return PublicExpectations(
        raw_files=len(paths),
        raw_rows=raw_rows,
        raw_customers=len(customers),
        raw_products=len(products),
        duplicate_rows=duplicate_rows,
        missing_measure_rows=missing_measures,
        missing_customer_rows=missing_customers,
        fact_rows=accepted,
        customer_rows=len(customers) + 1,
        product_rows=len(products),
        date_rows=(end - start).days + 1,
        revenue=round(revenue, 2),
        preserved_customer_ids=tuple(int(value) for value in schema["preserve_customer_ids"]),
    )


def _table_columns(connection: sqlite3.Connection) -> dict[str, set[str]]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
        ).fetchall()
    }
    return {
        table: {
            str(row[1]) for row in connection.execute(f'pragma table_info("{table}")').fetchall()
        }
        for table in tables
    }


def _candidate_metrics(database: Path, expectations: PublicExpectations) -> dict[str, Any]:
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        tables = _table_columns(connection)
        schema_parts = [
            len(tables) == len(EXPECTED_TABLE_COLUMNS),
            *(
                tables.get(table) == columns
                for table, columns in sorted(EXPECTED_TABLE_COLUMNS.items())
            ),
        ]
        schema_score = sum(schema_parts) / len(schema_parts)
        required = set(EXPECTED_TABLE_COLUMNS)
        if not required.issubset(tables):
            return {"schema_score": schema_score}
        scalar = lambda sql: connection.execute(sql).fetchone()[0]
        counts = {
            "facts": int(scalar("select count(*) from fact_transactions")),
            "distinct_facts": int(
                scalar("select count(distinct transaction_id) from fact_transactions")
            ),
            "customers": int(scalar("select count(*) from dim_customers")),
            "products": int(scalar("select count(*) from dim_products")),
            "dates": int(scalar("select count(*) from dim_dates")),
        }
        revenue = float(
            scalar("select round(coalesce(sum(total_amount),0),2) from fact_transactions")
        )
        average = float(
            scalar("select round(coalesce(avg(total_amount),0),2) from fact_transactions")
        )
        date_min, date_max = connection.execute(
            "select min(date_key),max(date_key) from fact_transactions"
        ).fetchone()
        fk_failures = len(connection.execute("pragma foreign_key_check").fetchall())
        unresolved = int(
            scalar(
                "select count(*) from fact_transactions f "
                "left join dim_customers c on c.customer_key=f.customer_key "
                "left join dim_products p on p.product_key=f.product_key "
                "left join dim_dates d on d.date_key=f.date_key "
                "where c.customer_key is null or p.product_key is null or d.date_key is null"
            )
        )
        formula = int(
            scalar(
                "select count(*) from fact_transactions where "
                "abs(total_amount-quantity*unit_price*(1-discount_pct))>0.011"
            )
        )
        nulls = int(
            scalar(
                "select count(*) from fact_transactions where quantity is null or "
                "unit_price is null or transaction_id is null"
            )
        )
        invalid_segment = int(
            scalar(
                "select count(*) from dim_customers where segment is null or "
                f"segment not in {tuple(sorted(_SEGMENTS))}"
            )
        )
        invalid_country = int(
            scalar(
                "select count(*) from dim_customers where country_code is null or "
                f"country_code not in {tuple(sorted(_COUNTRIES))}"
            )
        )
        invalid_supplier = int(
            scalar(
                "select count(*) from dim_products where supplier is null or "
                f"supplier not in {tuple(sorted(_SUPPLIERS))}"
            )
        )
        invalid_boolean = int(
            scalar("select count(*) from dim_products where is_active not in (0,1)")
        )
        empty_category = int(
            scalar("select count(*) from dim_products where category is null or trim(category)='' ")
        )
        unknown_rows = int(scalar("select count(*) from dim_customers where customer_id=0"))
        unknown_facts = int(
            scalar(
                "select count(*) from fact_transactions f join dim_customers c "
                "on c.customer_key=f.customer_key where c.customer_id=0"
            )
        )
        preserved = tuple(
            int(row[0])
            for row in connection.execute(
                "select customer_id from dim_customers where customer_id>=9001 order by 1"
            ).fetchall()
        )
        unique_customers = int(scalar("select count(distinct customer_key) from fact_transactions"))
        unique_products = int(scalar("select count(distinct product_key) from fact_transactions"))
        grouped = lambda sql: dict(connection.execute(sql).fetchall())
        channels = grouped(
            "select channel,count(*) from fact_transactions group by channel order by 1"
        )
        segments = grouped("select segment,count(*) from dim_customers group by segment order by 1")
        countries = grouped(
            "select country_code,count(*) from dim_customers group by country_code order by 1"
        )
        categories = grouped(
            "select category,count(*) from dim_products group by category order by 1"
        )
        active = int(scalar("select count(*) from dim_products where is_active=1"))
        date_dimension = connection.execute(
            "select date_key,full_date,year,month,day,day_of_week,is_weekend,quarter "
            "from dim_dates order by date_key"
        ).fetchall()
    start = date(2025, 1, 1)
    calendar_valid = len(date_dimension) == expectations.date_rows
    if calendar_valid:
        for index, row in enumerate(date_dimension):
            expected = start + timedelta(days=index)
            if tuple(row) != (
                int(expected.strftime("%Y%m%d")),
                expected.isoformat(),
                expected.year,
                expected.month,
                expected.day,
                expected.strftime("%A"),
                int(expected.weekday() >= 5),
                (expected.month - 1) // 3 + 1,
            ):
                calendar_valid = False
                break
    return {
        "schema_score": schema_score,
        **counts,
        "revenue": revenue,
        "average": average,
        "date_min": int(date_min) if date_min is not None else 0,
        "date_max": int(date_max) if date_max is not None else 0,
        "fk_failures": fk_failures,
        "unresolved": unresolved,
        "formula": formula,
        "nulls": nulls,
        "invalid_segment": invalid_segment,
        "invalid_country": invalid_country,
        "invalid_supplier": invalid_supplier,
        "invalid_boolean": invalid_boolean,
        "empty_category": empty_category,
        "unknown_rows": unknown_rows,
        "unknown_facts": unknown_facts,
        "preserved": preserved,
        "calendar_valid": calendar_valid,
        "unique_customers": unique_customers,
        "unique_products": unique_products,
        "channels": channels,
        "segments": segments,
        "countries": countries,
        "categories": categories,
        "active": active,
    }


def _expected_sidecars(
    expectations: PublicExpectations, metrics: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = {
        "extraction": {
            "raw_transaction_files": expectations.raw_files,
            "raw_transaction_rows": expectations.raw_rows,
            "raw_customer_records": expectations.raw_customers,
            "raw_product_records": expectations.raw_products,
        },
        "transformations": {
            "duplicates_removed": expectations.duplicate_rows,
            "null_rows_dropped": expectations.missing_measure_rows,
            "null_customer_ids_mapped": expectations.missing_customer_rows,
            "timestamps_standardized": True,
            "schema_drift_columns_filled": ["channel", "discount_pct"],
            "country_codes_standardized": metrics["invalid_country"] == 0,
            "supplier_names_standardized": metrics["invalid_supplier"] == 0,
            "boolean_fields_normalized": metrics["invalid_boolean"] == 0,
            "empty_categories_labeled": metrics["empty_category"] == 0,
        },
        "load": {
            "fact_transactions_rows": metrics["facts"],
            "dim_customers_rows": metrics["customers"],
            "dim_products_rows": metrics["products"],
            "dim_dates_rows": metrics["dates"],
            "referential_integrity_violations_remaining": metrics["unresolved"],
        },
        "quality_checks": {
            "no_duplicate_transaction_ids": metrics["facts"] == metrics["distinct_facts"],
            "no_null_quantities": metrics["nulls"] == 0,
            "no_null_unit_prices": metrics["nulls"] == 0,
            "total_amount_formula_correct": metrics["formula"] == 0,
            "all_foreign_keys_resolve": metrics["unresolved"] == 0,
            "all_segments_valid": metrics["invalid_segment"] == 0,
            "all_country_codes_iso2": metrics["invalid_country"] == 0,
        },
    }
    summary = {
        "fact_table": {
            "row_count": metrics["facts"],
            "unique_transactions": metrics["distinct_facts"],
            "unique_customers": metrics["unique_customers"],
            "unique_products": metrics["unique_products"],
            "total_revenue": metrics["revenue"],
            "avg_transaction_value": metrics["average"],
            "date_range": {"min": metrics["date_min"], "max": metrics["date_max"]},
            "channel_distribution": metrics["channels"],
        },
        "dim_customers": {
            "row_count": metrics["customers"],
            "segment_distribution": metrics["segments"],
            "country_distribution": metrics["countries"],
        },
        "dim_products": {
            "row_count": metrics["products"],
            "category_distribution": metrics["categories"],
            "active_products": metrics["active"],
        },
    }
    return report, summary


def inspect_candidate(public_root: Path, output_root: Path) -> WarehouseEvaluation:
    """Return ten continuous semantic scores and six exact terminal criteria."""
    expectations = _public_expectations(public_root)
    try:
        files = {
            path.relative_to(output_root).as_posix() for path in iter_regular_files(output_root)
        }
    except (OSError, ValueError):
        files = set()
    required = {"warehouse.db", "data_quality_report.json", "warehouse_summary.json"}
    scope = files == required
    readable_parts = 0
    report: object = None
    summary: object = None
    metrics: dict[str, Any] = {"schema_score": 0.0}
    reasons: list[str] = []
    try:
        report = json.loads((output_root / "data_quality_report.json").read_bytes())
        readable_parts += 1
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        reasons.append(f"quality report unreadable: {type(error).__name__}: {error}")
    try:
        summary = json.loads((output_root / "warehouse_summary.json").read_bytes())
        readable_parts += 1
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        reasons.append(f"summary unreadable: {type(error).__name__}: {error}")
    try:
        metrics = _candidate_metrics(output_root / "warehouse.db", expectations)
        readable_parts += 1
    except (KeyError, OSError, sqlite3.Error, TypeError, ValueError) as error:
        reasons.append(f"database unreadable: {type(error).__name__}: {error}")
    readable = readable_parts / 3
    schema_score = float(metrics.get("schema_score", 0.0))
    complete = "facts" in metrics
    if complete:
        row_score = (
            sum(
                (
                    _closeness(metrics["facts"], expectations.fact_rows, 0.05),
                    _closeness(metrics["customers"], expectations.customer_rows, 0.05),
                    _closeness(metrics["products"], expectations.product_rows, 0.05),
                    float(metrics["dates"] == expectations.date_rows),
                )
            )
            / 4
        )
        quality_score = (
            sum(
                float(value)
                for value in (
                    metrics["facts"] == metrics["distinct_facts"],
                    metrics["nulls"] == 0,
                    metrics["formula"] == 0,
                    metrics["unresolved"] == 0,
                    metrics["fk_failures"] == 0,
                )
            )
            / 5
        )
        standardization = (
            sum(
                float(value == 0)
                for value in (
                    metrics["invalid_segment"],
                    metrics["invalid_country"],
                    metrics["invalid_supplier"],
                    metrics["invalid_boolean"],
                    metrics["empty_category"],
                )
            )
            / 5
        )
        dimensions = (
            sum(
                (
                    _closeness(metrics["customers"], expectations.customer_rows, 0.05),
                    _closeness(metrics["products"], expectations.product_rows, 0.05),
                    float(metrics["calendar_valid"]),
                    float(metrics["unknown_rows"] == 1),
                    float(metrics["preserved"] == expectations.preserved_customer_ids),
                )
            )
            / 5
        )
        relations = (
            sum(
                (
                    float(metrics["unresolved"] == 0),
                    float(metrics["fk_failures"] == 0),
                    float(metrics["unknown_facts"] == expectations.missing_customer_rows),
                )
            )
            / 3
        )
        revenue_score = _closeness(metrics["revenue"], expectations.revenue, 0.02)
        expected_report, expected_summary = _expected_sidecars(expectations, metrics)
        quality_report = float(report == expected_report)
        summary_score = float(summary == expected_summary)
    else:
        row_score = quality_score = standardization = dimensions = relations = 0.0
        revenue_score = quality_report = summary_score = 0.0
    terminal = {
        "data_quality_checks": quality_score == 1.0,
        "revenue_within_tolerance": complete
        and abs(metrics["revenue"] - expectations.revenue)
        <= max(abs(expectations.revenue) * 0.02, 1e-12),
        "row_counts_within_tolerance": row_score == 1.0,
        "schema_correct": bool(
            schema_score == 1.0
            and complete
            and metrics["calendar_valid"]
            and metrics["unknown_rows"] == 1
            and metrics["preserved"] == expectations.preserved_customer_ids
        ),
        "sidecars_truthful": quality_report == 1.0 and summary_score == 1.0,
        "standardization_correct": standardization == 1.0,
    }
    scores = {
        "aggregates": revenue_score,
        "dimensions": dimensions,
        "exact": float(scope and all(terminal.values())),
        "facts": row_score,
        "normalization": standardization,
        "quality_report": quality_report,
        "readable": readable,
        "relations": relations,
        "schema": schema_score,
        "summary": summary_score,
    }
    evidence = content_id(
        "v72-warehouse-recovery-evidence",
        {
            "expectations": {
                **expectations.__dict__,
                "preserved_customer_ids": list(expectations.preserved_customer_ids),
            },
            "scores": scores,
            "terminal": terminal,
            "scope": scope,
            "reason": reasons,
        },
    )
    return WarehouseEvaluation(
        scores=scores,
        terminal_criteria=terminal,
        output_scope_valid=scope,
        evidence_record_id=evidence,
        reason="; ".join(reasons) or "semantic warehouse checks completed",
    )


def verify_warehouse_etl(
    *,
    task_id: str,
    reward_contract: RewardContract,
    public_input_root: Path,
    sealed_output_root: Path,
) -> RewardResult:
    """Recompute the final vector from public sources and sealed candidate outputs."""
    if {component.component_id for component in reward_contract.components} != _COMPONENTS:
        return RewardResult.from_system_failure(
            task_id=task_id,
            reward_contract_id=reward_contract.contract_id,
            status="task_defect",
            failure_detail="warehouse reward contract has the wrong component set",
        )
    if set(reward_contract.hard_gate_public_rule_ids) != {"rule.output-scope"}:
        return RewardResult.from_system_failure(
            task_id=task_id,
            reward_contract_id=reward_contract.contract_id,
            status="task_defect",
            failure_detail="warehouse reward contract has the wrong hard gate",
        )
    try:
        evaluation = inspect_candidate(public_input_root, sealed_output_root)
    except (csv.Error, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        return RewardResult.from_system_failure(
            task_id=task_id,
            reward_contract_id=reward_contract.contract_id,
            status="task_defect",
            failure_detail=f"public warehouse authority is invalid: {type(error).__name__}: {error}",
        )
    values = {
        "reward.data-quality": evaluation.scores["relations"],
        "reward.revenue": evaluation.scores["aggregates"],
        "reward.row-counts": evaluation.scores["facts"],
        "reward.schema": (evaluation.scores["schema"] + evaluation.scores["dimensions"]) / 2,
        "reward.sidecars": (evaluation.scores["quality_report"] + evaluation.scores["summary"]) / 2,
        "reward.standardization": evaluation.scores["normalization"],
    }
    return RewardResult.from_policy_result(
        task_id=task_id,
        contract=reward_contract,
        component_values=values,
        hard_gate_values={"rule.output-scope": evaluation.output_scope_valid},
        evidence_record_ids=(evaluation.evidence_record_id,),
    )


def run_verifier_process(payload: dict[str, object]) -> dict[str, object]:
    expected = {"task_id", "reward_contract", "public_input_root", "sealed_output_root"}
    if set(payload) != expected or type(payload["reward_contract"]) is not dict:
        raise ValueError("verifier process payload is invalid")
    if any(
        type(payload[name]) is not str
        for name in ("task_id", "public_input_root", "sealed_output_root")
    ):
        raise ValueError("verifier process payload types are invalid")
    contract = RewardContract.model_validate_json(canonical_json(payload["reward_contract"]))
    return verify_warehouse_etl(
        task_id=payload["task_id"],
        reward_contract=contract,
        public_input_root=Path(payload["public_input_root"]),
        sealed_output_root=Path(payload["sealed_output_root"]),
    ).model_dump(mode="json")


__all__ = ["WarehouseEvaluation", "inspect_candidate", "verify_warehouse_etl"]
