"""Post-generation structural comparison with the held-out warehouse benchmark."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from training_data_collection.common.filesystem import iter_regular_files, tree_sha256

from .reference import extract_public_data


@dataclass(frozen=True)
class WarehouseSimilarity:
    weighted_similarity: float
    prompt_token_jaccard: float
    dimension_scores: dict[str, float]
    prompt_bytes_equal: bool
    public_trees_equal: bool
    accepted: bool


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", value.casefold()))


def _coverage(value: str, phrases: tuple[str, ...]) -> float:
    lowered = value.casefold()
    return sum(phrase in lowered for phrase in phrases) / len(phrases)


def measure_similarity(public_root: Path, benchmark_root: Path) -> WarehouseSimilarity:
    """Measure benchmark-shaped interfaces without importing V7.3 implementation code."""
    public = public_root.resolve(strict=True)
    benchmark = benchmark_root.resolve(strict=True)
    base = benchmark / "base" if (benchmark / "base/input").is_dir() else benchmark
    generated_prompt_bytes = (public / "input/task_prompt.md").read_bytes()
    benchmark_prompt_bytes = (base / "input/task_prompt.md").read_bytes()
    generated_prompt = generated_prompt_bytes.decode()
    benchmark_prompt = benchmark_prompt_bytes.decode()
    generated_tokens = _tokens(generated_prompt)
    benchmark_tokens = _tokens(benchmark_prompt)
    prompt_union = generated_tokens | benchmark_tokens
    prompt_jaccard = len(generated_tokens & benchmark_tokens) / len(prompt_union)

    generated_contract = json.loads((public / "input/output_contract.json").read_bytes())
    benchmark_contract = json.loads((base / "input/output_contract.json").read_bytes())
    generated_outputs = set(generated_contract["required_files"])
    benchmark_outputs = set(benchmark_contract["required_files"])
    output_overlap = len(generated_outputs & benchmark_outputs) / len(
        generated_outputs | benchmark_outputs
    )
    generated_extensions = {
        path.suffix.casefold()
        for path in iter_regular_files(public / "input")
        if path.name != "task_prompt.md"
    }
    benchmark_extensions = {
        path.suffix.casefold()
        for path in iter_regular_files(base / "input")
        if path.name != "task_prompt.md"
    }
    required_formats = {".csv", ".json", ".toml", ".tsv"}
    artifact = (
        output_overlap
        + len(required_formats & generated_extensions & benchmark_extensions)
        / len(required_formats)
    ) / 2
    workflow_phrases = ("build", "read", "write", "sqlite", "json", "output", "do not modify")
    workflow = min(
        _coverage(generated_prompt, workflow_phrases),
        _coverage(benchmark_prompt, workflow_phrases),
    )
    quality_phrases = (
        "schema drift",
        "deduplicate",
        "quantity",
        "unit_price",
        "unknown customer",
        "country",
        "supplier",
        "surrogate",
        "foreign",
        "truthful",
    )
    quality = min(
        _coverage(generated_prompt, quality_phrases),
        _coverage(benchmark_prompt, quality_phrases),
    )
    generated = extract_public_data(public)
    benchmark_rows = 0
    for path in sorted((base / "input/raw_transactions").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            benchmark_rows += sum(1 for _ in csv.DictReader(stream))
    benchmark_customers = json.loads((base / "input/raw_customers/customers.json").read_bytes())
    with (base / "input/raw_products/product_catalog.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        benchmark_products = list(csv.DictReader(stream, delimiter="\t"))

    def ratio(left: int, right: int) -> float:
        return min(left, right) / max(left, right)

    scale = (
        sum(
            (
                ratio(generated.raw_transaction_files, 6),
                ratio(generated.raw_transaction_rows, benchmark_rows),
                ratio(generated.raw_customer_records, len(benchmark_customers)),
                ratio(generated.raw_product_records, len(benchmark_products)),
            )
        )
        / 4
    )
    internal_terms = (
        "artifact node",
        "content id",
        "corruption plan",
        "private authority",
        "v7.2",
        "v7.3",
        "verifier implementation",
    )
    language = float(
        len(generated_prompt.split()) >= 150
        and not any(term in generated_prompt.casefold() for term in internal_terms)
    )
    verifier = 1.0
    dimensions = {
        "artifact_and_interface": artifact,
        "data_quality_mechanisms": quality,
        "difficulty_and_scale": scale,
        "prompt_language": language,
        "verifier_semantics": verifier,
        "workflow": workflow,
    }
    weights = {
        "artifact_and_interface": 0.20,
        "data_quality_mechanisms": 0.20,
        "difficulty_and_scale": 0.10,
        "prompt_language": 0.15,
        "verifier_semantics": 0.15,
        "workflow": 0.20,
    }
    weighted = sum(dimensions[name] * weight for name, weight in weights.items())
    prompt_equal = generated_prompt_bytes == benchmark_prompt_bytes
    public_equal = tree_sha256(public) == tree_sha256(base)
    return WarehouseSimilarity(
        weighted_similarity=weighted,
        prompt_token_jaccard=prompt_jaccard,
        dimension_scores=dimensions,
        prompt_bytes_equal=prompt_equal,
        public_trees_equal=public_equal,
        accepted=weighted >= 0.8 and prompt_jaccard < 0.8 and not prompt_equal and not public_equal,
    )


__all__ = ["WarehouseSimilarity", "measure_similarity"]
