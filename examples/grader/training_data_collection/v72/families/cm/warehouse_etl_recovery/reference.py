"""Public-only reference solver for the V7.2 warehouse recovery family."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from training_data_collection.common.filesystem import (
    iter_regular_files,
    prepare_empty_output_root,
    tree_sha256,
)

CUSTOMERS_PATH = "raw_customers/customers.json"
PRODUCTS_PATH = "raw_products/product_catalog.tsv"
RAW_TRANSACTIONS_PATH = "raw_transactions"
SCHEMA_PATH = "target_schema_spec.json"

_COUNTRIES = {
    "ca": "CA",
    "can": "CA",
    "canada": "CA",
    "de": "DE",
    "deutschland": "DE",
    "germany": "DE",
    "gb": "GB",
    "uk": "GB",
    "unitedkingdom": "GB",
    "unitedstates": "US",
    "us": "US",
    "usa": "US",
}
_SUPPLIERS = {
    "bluemesa": "Blue Mesa",
    "cedarworks": "Cedar Works",
    "harborwholesale": "Harbor Wholesale",
    "northstarsupply": "Northstar Supply",
}


@dataclass(frozen=True)
class ExtractedData:
    customers: tuple[dict[str, object], ...]
    products: tuple[dict[str, object], ...]
    transactions: tuple[dict[str, object], ...]
    raw_transaction_files: int
    raw_transaction_rows: int
    raw_customer_records: int
    raw_product_records: int
    duplicates_removed: int
    null_rows_dropped: int
    null_customer_ids_mapped: int
    schema_drift_columns_filled: tuple[str, ...]


def _token(value: object) -> str:
    return re.sub(r"[^a-z]", "", str(value or "").casefold())


def _timestamp(value: str) -> datetime:
    for pattern in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            pass
    raise ValueError(f"unsupported transaction timestamp: {value!r}")


def _boolean(value: object) -> int:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "y", "yes"}:
        return 1
    if normalized in {"0", "false", "n", "no"}:
        return 0
    raise ValueError(f"unsupported product boolean: {value!r}")


def extract_public_data(public_root: Path) -> ExtractedData:
    """Normalize mixed-format public sources without private-world access."""
    root = public_root.resolve(strict=True)
    if (root / "input").is_dir():
        root = (root / "input").resolve(strict=True)
    paths = tuple(sorted((root / RAW_TRANSACTIONS_PATH).glob("transactions_*.csv")))
    seen: set[str] = set()
    facts: list[dict[str, object]] = []
    raw_rows = duplicates = null_rows = null_customers = 0
    missing_columns: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing_columns.update({"channel", "discount_pct"} - fields)
            for row in reader:
                raw_rows += 1
                transaction_id = str(row.get("transaction_id", "")).strip()
                if not transaction_id:
                    raise ValueError("transaction identities must be nonempty")
                if transaction_id in seen:
                    duplicates += 1
                    continue
                seen.add(transaction_id)
                quantity = str(row.get("quantity", "")).strip()
                price = str(row.get("unit_price", "")).strip()
                if not quantity or not price:
                    null_rows += 1
                    continue
                raw_customer = str(row.get("customer_id", "")).strip()
                customer_id = int(float(raw_customer)) if raw_customer else 0
                null_customers += not bool(raw_customer)
                timestamp = _timestamp(str(row.get("timestamp", "")).strip())
                facts.append(
                    {
                        "transaction_id": transaction_id,
                        "customer_id": customer_id,
                        "product_id": int(float(str(row["product_id"]))),
                        "date_key": int(timestamp.strftime("%Y%m%d")),
                        "quantity": int(float(quantity)),
                        "unit_price": float(price),
                        "discount_pct": float(row.get("discount_pct") or 0.0),
                        "channel": str(row.get("channel") or "unknown").strip() or "unknown",
                    }
                )

    customer_source = json.loads((root / CUSTOMERS_PATH).read_bytes())
    if type(customer_source) is not list or any(type(row) is not dict for row in customer_source):
        raise ValueError("customer source must be an array of objects")
    customers = tuple(
        {
            "customer_id": int(row["id"]),
            "name": str(row.get("name") or f"Account {int(row['id'])}"),
            "email": str(row.get("email") or ""),
            "segment": (
                value
                if (value := str(row.get("segment") or "").strip().casefold())
                in {"basic", "premium", "standard"}
                else "unknown"
            ),
            "country_code": _COUNTRIES.get(_token(row.get("country")), "XX"),
            "registration_date": str(row.get("registration_date") or ""),
        }
        for row in customer_source
    )
    with (root / PRODUCTS_PATH).open(encoding="utf-8", newline="") as stream:
        product_source = list(csv.DictReader(stream, delimiter="\t"))
    products = tuple(
        {
            "product_id": int(row["product_id"]),
            "product_name": row["product_name"],
            "category": row["category"].strip() or "Uncategorized",
            "subcategory": row["subcategory"],
            "base_price": float(row["base_price"]),
            "cost_price": float(row["cost_price"]),
            "supplier": _SUPPLIERS[_token(row["supplier"])],
            "is_active": _boolean(row["is_active"]),
        }
        for row in product_source
    )
    return ExtractedData(
        customers=customers,
        products=products,
        transactions=tuple(facts),
        raw_transaction_files=len(paths),
        raw_transaction_rows=raw_rows,
        raw_customer_records=len(customer_source),
        raw_product_records=len(product_source),
        duplicates_removed=duplicates,
        null_rows_dropped=null_rows,
        null_customer_ids_mapped=null_customers,
        schema_drift_columns_filled=tuple(sorted(missing_columns)),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _database_metrics(path: Path) -> dict[str, object]:
    with sqlite3.connect(path) as connection:
        scalar = lambda sql: connection.execute(sql).fetchone()[0]
        return {
            "facts": int(scalar("select count(*) from fact_transactions")),
            "transactions": int(
                scalar("select count(distinct transaction_id) from fact_transactions")
            ),
            "customers": int(scalar("select count(*) from dim_customers")),
            "products": int(scalar("select count(*) from dim_products")),
            "dates": int(scalar("select count(*) from dim_dates")),
            "unique_customers": int(
                scalar("select count(distinct customer_key) from fact_transactions")
            ),
            "unique_products": int(
                scalar("select count(distinct product_key) from fact_transactions")
            ),
            "revenue": float(
                scalar("select round(coalesce(sum(total_amount), 0), 2) from fact_transactions")
            ),
            "average": float(
                scalar("select round(coalesce(avg(total_amount), 0), 2) from fact_transactions")
            ),
            "date_min": int(scalar("select min(date_key) from fact_transactions")),
            "date_max": int(scalar("select max(date_key) from fact_transactions")),
            "channels": dict(
                connection.execute(
                    "select channel, count(*) from fact_transactions group by channel order by 1"
                ).fetchall()
            ),
            "segments": dict(
                connection.execute(
                    "select segment, count(*) from dim_customers group by segment order by 1"
                ).fetchall()
            ),
            "countries": dict(
                connection.execute(
                    "select country_code, count(*) from dim_customers "
                    "group by country_code order by 1"
                ).fetchall()
            ),
            "categories": dict(
                connection.execute(
                    "select category, count(*) from dim_products group by category order by 1"
                ).fetchall()
            ),
            "active": int(scalar("select count(*) from dim_products where is_active = 1")),
        }


def _sidecars(extracted: ExtractedData, metrics: dict[str, object]) -> tuple[dict, dict]:
    report = {
        "extraction": {
            "raw_transaction_files": extracted.raw_transaction_files,
            "raw_transaction_rows": extracted.raw_transaction_rows,
            "raw_customer_records": extracted.raw_customer_records,
            "raw_product_records": extracted.raw_product_records,
        },
        "transformations": {
            "duplicates_removed": extracted.duplicates_removed,
            "null_rows_dropped": extracted.null_rows_dropped,
            "null_customer_ids_mapped": extracted.null_customer_ids_mapped,
            "timestamps_standardized": True,
            "schema_drift_columns_filled": list(extracted.schema_drift_columns_filled),
            "country_codes_standardized": True,
            "supplier_names_standardized": True,
            "boolean_fields_normalized": True,
            "empty_categories_labeled": True,
        },
        "load": {
            "fact_transactions_rows": metrics["facts"],
            "dim_customers_rows": metrics["customers"],
            "dim_products_rows": metrics["products"],
            "dim_dates_rows": metrics["dates"],
            "referential_integrity_violations_remaining": 0,
        },
        "quality_checks": {
            "no_duplicate_transaction_ids": metrics["facts"] == metrics["transactions"],
            "no_null_quantities": True,
            "no_null_unit_prices": True,
            "total_amount_formula_correct": True,
            "all_foreign_keys_resolve": True,
            "all_segments_valid": True,
            "all_country_codes_iso2": True,
        },
    }
    summary = {
        "fact_table": {
            "row_count": metrics["facts"],
            "unique_transactions": metrics["transactions"],
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


def solve_reference(public_root: Path, output_root: Path) -> None:
    """Build the exact warehouse bundle from solver-visible public files."""
    root = public_root.resolve(strict=True)
    schema = json.loads((root / SCHEMA_PATH).read_bytes())
    extracted = extract_public_data(root)
    prepare_empty_output_root(output_root)
    database = output_root / "warehouse.db"
    customers = (
        {
            "customer_id": 0,
            "name": "Unknown Customer",
            "email": "",
            "segment": "unknown",
            "country_code": "XX",
            "registration_date": "",
        },
        *extracted.customers,
    )
    ordered_customers = sorted(customers, key=lambda row: int(row["customer_id"]))
    ordered_products = sorted(extracted.products, key=lambda row: int(row["product_id"]))
    customer_keys = {int(row["customer_id"]): i for i, row in enumerate(ordered_customers, 1)}
    product_keys = {int(row["product_id"]): i for i, row in enumerate(ordered_products, 1)}
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "pragma foreign_keys=on;"
            "create table dim_customers (customer_key integer primary key, customer_id integer "
            "not null unique, name text not null, email text not null, segment text not null, "
            "country_code text not null, registration_date text not null);"
            "create table dim_products (product_key integer primary key, product_id integer not "
            "null unique, product_name text not null, category text not null, subcategory text "
            "not null, base_price real not null, cost_price real not null, supplier text not null, "
            "is_active integer not null);"
            "create table dim_dates (date_key integer primary key, full_date text not null, year "
            "integer not null, month integer not null, day integer not null, day_of_week text not "
            "null, is_weekend integer not null, quarter integer not null);"
            "create table fact_transactions (transaction_id text primary key, customer_key integer "
            "not null references dim_customers, product_key integer not null references "
            "dim_products, date_key integer not null references dim_dates, quantity integer not "
            "null, unit_price real not null, total_amount real not null, discount_pct real not null, "
            "channel text not null);"
        )
        connection.executemany(
            "insert into dim_customers values (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    customer_keys[int(row["customer_id"])],
                    row["customer_id"],
                    row["name"],
                    row["email"],
                    row["segment"],
                    row["country_code"],
                    row["registration_date"],
                )
                for row in ordered_customers
            ],
        )
        connection.executemany(
            "insert into dim_products values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    product_keys[int(row["product_id"])],
                    row["product_id"],
                    row["product_name"],
                    row["category"],
                    row["subcategory"],
                    row["base_price"],
                    row["cost_price"],
                    row["supplier"],
                    row["is_active"],
                )
                for row in ordered_products
            ],
        )
        start = date.fromisoformat(schema["date_dimension"]["start"])
        end = date.fromisoformat(schema["date_dimension"]["end"])
        dates = []
        current = start
        while current <= end:
            dates.append(
                (
                    int(current.strftime("%Y%m%d")),
                    current.isoformat(),
                    current.year,
                    current.month,
                    current.day,
                    current.strftime("%A"),
                    int(current.weekday() >= 5),
                    (current.month - 1) // 3 + 1,
                )
            )
            current += timedelta(days=1)
        connection.executemany("insert into dim_dates values (?, ?, ?, ?, ?, ?, ?, ?)", dates)
        connection.executemany(
            "insert into fact_transactions values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["transaction_id"],
                    customer_keys[int(row["customer_id"])],
                    product_keys[int(row["product_id"])],
                    row["date_key"],
                    row["quantity"],
                    row["unit_price"],
                    round(
                        int(row["quantity"])
                        * float(row["unit_price"])
                        * (1.0 - float(row["discount_pct"])),
                        2,
                    ),
                    row["discount_pct"],
                    row["channel"],
                )
                for row in extracted.transactions
            ],
        )
    metrics = _database_metrics(database)
    report, summary = _sidecars(extracted, metrics)
    _write_json(output_root / "data_quality_report.json", report)
    _write_json(output_root / "warehouse_summary.json", summary)


def run_reference_process(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {"public_input_root", "output_root"} or any(
        type(payload[name]) is not str for name in payload
    ):
        raise ValueError("reference process payload is invalid")
    output_root = Path(payload["output_root"])
    solve_reference(Path(payload["public_input_root"]), output_root)
    return {
        "output_tree_sha256": tree_sha256(output_root),
        "output_file_count": len(iter_regular_files(output_root)),
    }


__all__ = ["ExtractedData", "extract_public_data", "solve_reference"]
