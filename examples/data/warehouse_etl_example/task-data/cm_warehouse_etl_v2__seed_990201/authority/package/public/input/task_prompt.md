# Retail Warehouse ETL Recovery

You are working on a Linux VM with the `/bin/sh` shell. Tools: `python3`, `sqlite3`. Network: disabled.

## Goal

Build a clean star-schema SQLite warehouse from six months of messy retail data and produce two truthful JSON sidecars.

The staged transaction partitions, customer records, and product catalog contain schema drift, duplicates, missing values, and categorical variants. Reconcile them using the supplied schema and output contracts.

## Visible Inputs (read-only)

- `input/raw_customers/customers.json`: Customer JSON records with country, segment, and provisional-record drift.
- `input/output_contract.json`: Exact required filenames and JSON sidecar section and field contract.
- `input/raw_products/product_catalog.tsv`: Product TSV records with supplier, boolean, and category drift.
- `input/runtime_env`: Offline Python runtime manifest.
- `input/target_schema_spec.json`: Authoritative star-schema tables, columns, value domains, date range, and customer preservation rules.
- `input/raw_transactions`: Six monthly transaction CSV partitions with schema drift, duplicates, missing measures, missing customer IDs, and several timestamp formats.
- `software/README.txt`: Local Python and SQLite usage notes.

## What to do

1. Normalize segments to basic, premium, standard, or unknown; countries to CA, DE, GB, US, or XX; suppliers to the published canonical names; activity flags to 0 or 1; and blank product categories to Uncategorized. Preserve every listed provisional customer with documented fallbacks and add exactly one Unknown Customer row with customer_id 0.
2. Read all six monthly CSV files, fill missing discount_pct with 0 and channel with unknown, normalize all published timestamp formats, keep the first row for each duplicate transaction_id, drop rows missing quantity or unit_price, and map blank customer IDs to customer_id 0.
3. Create deterministic customer and product surrogate keys, derive YYYYMMDD date keys, resolve every accepted fact to all dimensions, and recompute total_amount as quantity times unit_price times one minus discount_pct rounded to cents.
4. Write exactly warehouse.db, data_quality_report.json, and warehouse_summary.json directly under output. Do not modify input or software.
5. Build every date from 2025-01-01 through 2025-06-30 inclusive, enforce unique transaction identities, keep required measures non-null and finite, preserve formula correctness, and leave no unresolved foreign keys.
6. Populate every field listed in output_contract.json. Recompute extraction and defect counts from the staged sources and all load counts, quality flags, distributions, date bounds, averages, and revenue from the final database so both sidecars remain complete and truthful.

- Authority precedence: target_schema_spec.json overrides raw spelling and representation variants.

## Required Output (write under `output/`)

- `output/data_quality_report.json`: Truthful extraction, transformation, load, and quality-check evidence matching the source files and final database.
- `output/warehouse.db`: SQLite star-schema warehouse with three dimensions and one fact table.
- `output/warehouse_summary.json`: Truthful fact and dimension aggregates recomputed from warehouse.db.

Use this prompt and the staged input contracts as the complete requirements for the deliverables.
Permitted dependencies: `python`, `sqlite`.
Write only these deliverables under `output/`.
Do not modify files under `input/` or `software/`.
