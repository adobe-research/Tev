"""Structurally independent public-input solver for warehouse ETL recovery."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from training_data_collection.common.filesystem import (
    iter_regular_files,
    prepare_empty_output_root,
    tree_sha256,
)


def _canonical_token(value: object) -> str:
    return re.sub("[^a-z]", "", str(value or "").lower())


def _parse_time(value: str) -> datetime:
    attempts = ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M:%S")
    for attempt in attempts:
        try:
            return datetime.strptime(value, attempt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unknown timestamp representation: {value}")


def _dump(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def solve_alternative(public_root: Path, output_root: Path) -> None:
    """Solve from raw records using an implementation independent of the reference module."""
    root = public_root.resolve(strict=True)
    if (root / "input").is_dir():
        root = (root / "input").resolve(strict=True)
    schema = json.loads((root / "target_schema_spec.json").read_bytes())
    customers_source = json.loads((root / "raw_customers/customers.json").read_bytes())
    country_lookup = {
        "ca": "CA",
        "can": "CA",
        "canada": "CA",
        "de": "DE",
        "deutschland": "DE",
        "germany": "DE",
        "gb": "GB",
        "uk": "GB",
        "unitedkingdom": "GB",
        "us": "US",
        "usa": "US",
        "unitedstates": "US",
    }
    normalized_customers = []
    for item in customers_source:
        identity = int(item["id"])
        segment = str(item.get("segment") or "").strip().lower()
        normalized_customers.append(
            (
                identity,
                str(item.get("name") or f"Account {identity}"),
                str(item.get("email") or ""),
                segment if segment in {"basic", "premium", "standard"} else "unknown",
                country_lookup.get(_canonical_token(item.get("country")), "XX"),
                str(item.get("registration_date") or ""),
            )
        )
    normalized_customers.append((0, "Unknown Customer", "", "unknown", "XX", ""))
    normalized_customers.sort()

    supplier_lookup = {
        "bluemesa": "Blue Mesa",
        "cedarworks": "Cedar Works",
        "harborwholesale": "Harbor Wholesale",
        "northstarsupply": "Northstar Supply",
    }
    with (root / "raw_products/product_catalog.tsv").open(encoding="utf-8", newline="") as stream:
        raw_products = list(csv.DictReader(stream, delimiter="\t"))
    normalized_products = []
    for item in raw_products:
        enabled = str(item["is_active"]).strip().lower()
        if enabled not in {"0", "1", "false", "true", "n", "no", "y", "yes"}:
            raise ValueError("product active flag is invalid")
        normalized_products.append(
            (
                int(item["product_id"]),
                item["product_name"],
                item["category"].strip() or "Uncategorized",
                item["subcategory"],
                float(item["base_price"]),
                float(item["cost_price"]),
                supplier_lookup[_canonical_token(item["supplier"])],
                int(enabled in {"1", "true", "y", "yes"}),
            )
        )
    normalized_products.sort()

    partitions = sorted((root / "raw_transactions").glob("transactions_*.csv"))
    observed: set[str] = set()
    raw_count = duplicate_count = missing_measure_count = missing_customer_count = 0
    drift: set[str] = set()
    normalized_facts = []
    for partition in partitions:
        with partition.open(encoding="utf-8", newline="") as stream:
            rows = csv.DictReader(stream)
            drift.update({"channel", "discount_pct"} - set(rows.fieldnames or ()))
            for item in rows:
                raw_count += 1
                identity = str(item.get("transaction_id") or "").strip()
                if identity in observed:
                    duplicate_count += 1
                    continue
                observed.add(identity)
                quantity_text = str(item.get("quantity") or "").strip()
                price_text = str(item.get("unit_price") or "").strip()
                if not quantity_text or not price_text:
                    missing_measure_count += 1
                    continue
                customer_text = str(item.get("customer_id") or "").strip()
                if not customer_text:
                    missing_customer_count += 1
                quantity = int(float(quantity_text))
                unit_price = float(price_text)
                discount = float(item.get("discount_pct") or 0)
                normalized_facts.append(
                    (
                        identity,
                        int(float(customer_text)) if customer_text else 0,
                        int(float(item["product_id"])),
                        int(_parse_time(str(item["timestamp"])).strftime("%Y%m%d")),
                        quantity,
                        unit_price,
                        round(quantity * unit_price * (1 - discount), 2),
                        discount,
                        str(item.get("channel") or "unknown").strip() or "unknown",
                    )
                )

    prepare_empty_output_root(output_root)
    database = output_root / "warehouse.db"
    customer_keys = {row[0]: key for key, row in enumerate(normalized_customers, 1)}
    product_keys = {row[0]: key for key, row in enumerate(normalized_products, 1)}
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "pragma foreign_keys=on;"
            "create table dim_customers(customer_key integer primary key, customer_id integer "
            "not null unique, name text not null, email text not null, segment text not null, "
            "country_code text not null, registration_date text not null);"
            "create table dim_products(product_key integer primary key, product_id integer not "
            "null unique, product_name text not null, category text not null, subcategory text not "
            "null, base_price real not null, cost_price real not null, supplier text not null, "
            "is_active integer not null);"
            "create table dim_dates(date_key integer primary key, full_date text not null, year "
            "integer not null, month integer not null, day integer not null, day_of_week text not "
            "null, is_weekend integer not null, quarter integer not null);"
            "create table fact_transactions(transaction_id text primary key, customer_key integer "
            "not null references dim_customers, product_key integer not null references "
            "dim_products, date_key integer not null references dim_dates, quantity integer not "
            "null, unit_price real not null, total_amount real not null, discount_pct real not null, "
            "channel text not null);"
        )
        connection.executemany(
            "insert into dim_customers values (?, ?, ?, ?, ?, ?, ?)",
            [(customer_keys[row[0]], *row) for row in normalized_customers],
        )
        connection.executemany(
            "insert into dim_products values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(product_keys[row[0]], *row) for row in normalized_products],
        )
        first = date.fromisoformat(schema["date_dimension"]["start"])
        last = date.fromisoformat(schema["date_dimension"]["end"])
        calendar = []
        cursor = first
        while cursor <= last:
            calendar.append(
                (
                    int(cursor.strftime("%Y%m%d")),
                    cursor.isoformat(),
                    cursor.year,
                    cursor.month,
                    cursor.day,
                    cursor.strftime("%A"),
                    int(cursor.weekday() > 4),
                    (cursor.month + 2) // 3,
                )
            )
            cursor += timedelta(days=1)
        connection.executemany("insert into dim_dates values (?, ?, ?, ?, ?, ?, ?, ?)", calendar)
        connection.executemany(
            "insert into fact_transactions values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row[0],
                    customer_keys[row[1]],
                    product_keys[row[2]],
                    *row[3:],
                )
                for row in normalized_facts
            ],
        )
        one = lambda sql: connection.execute(sql).fetchone()[0]
        grouped = lambda sql: dict(connection.execute(sql).fetchall())
        metrics = {
            "facts": int(one("select count(*) from fact_transactions")),
            "unique_transactions": int(
                one("select count(distinct transaction_id) from fact_transactions")
            ),
            "unique_customers": int(
                one("select count(distinct customer_key) from fact_transactions")
            ),
            "unique_products": int(
                one("select count(distinct product_key) from fact_transactions")
            ),
            "revenue": float(
                one("select round(coalesce(sum(total_amount),0),2) from fact_transactions")
            ),
            "average": float(
                one("select round(coalesce(avg(total_amount),0),2) from fact_transactions")
            ),
            "date_min": int(one("select min(date_key) from fact_transactions")),
            "date_max": int(one("select max(date_key) from fact_transactions")),
            "channels": grouped(
                "select channel,count(*) from fact_transactions group by channel order by 1"
            ),
            "segments": grouped(
                "select segment,count(*) from dim_customers group by segment order by 1"
            ),
            "countries": grouped(
                "select country_code,count(*) from dim_customers group by country_code order by 1"
            ),
            "categories": grouped(
                "select category,count(*) from dim_products group by category order by 1"
            ),
            "active": int(one("select count(*) from dim_products where is_active=1")),
        }

    report = {
        "extraction": {
            "raw_transaction_files": len(partitions),
            "raw_transaction_rows": raw_count,
            "raw_customer_records": len(customers_source),
            "raw_product_records": len(raw_products),
        },
        "transformations": {
            "duplicates_removed": duplicate_count,
            "null_rows_dropped": missing_measure_count,
            "null_customer_ids_mapped": missing_customer_count,
            "timestamps_standardized": True,
            "schema_drift_columns_filled": sorted(drift),
            "country_codes_standardized": True,
            "supplier_names_standardized": True,
            "boolean_fields_normalized": True,
            "empty_categories_labeled": True,
        },
        "load": {
            "fact_transactions_rows": metrics["facts"],
            "dim_customers_rows": len(normalized_customers),
            "dim_products_rows": len(normalized_products),
            "dim_dates_rows": len(calendar),
            "referential_integrity_violations_remaining": 0,
        },
        "quality_checks": {
            "no_duplicate_transaction_ids": True,
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
            "unique_transactions": metrics["unique_transactions"],
            "unique_customers": metrics["unique_customers"],
            "unique_products": metrics["unique_products"],
            "total_revenue": metrics["revenue"],
            "avg_transaction_value": metrics["average"],
            "date_range": {"min": metrics["date_min"], "max": metrics["date_max"]},
            "channel_distribution": metrics["channels"],
        },
        "dim_customers": {
            "row_count": len(normalized_customers),
            "segment_distribution": metrics["segments"],
            "country_distribution": metrics["countries"],
        },
        "dim_products": {
            "row_count": len(normalized_products),
            "category_distribution": metrics["categories"],
            "active_products": metrics["active"],
        },
    }
    _dump(output_root / "data_quality_report.json", report)
    _dump(output_root / "warehouse_summary.json", summary)


def run_alternative_process(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {"public_input_root", "output_root"} or any(
        type(payload[name]) is not str for name in payload
    ):
        raise ValueError("alternative process payload is invalid")
    output = Path(payload["output_root"])
    solve_alternative(Path(payload["public_input_root"]), output)
    return {
        "output_tree_sha256": tree_sha256(output),
        "output_file_count": len(iter_regular_files(output)),
    }


__all__ = ["solve_alternative"]
