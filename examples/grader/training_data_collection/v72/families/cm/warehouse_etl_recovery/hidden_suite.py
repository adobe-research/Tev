"""Seed-separated executable hidden workspaces for warehouse qualification."""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from training_data_collection.common.filesystem import (
    atomic_directory,
    iter_regular_files,
    tree_sha256,
)
from training_data_collection.common.identity import content_digest, derive_seed
from training_data_collection.v72.contracts.artifact import ArtifactGraph
from training_data_collection.v72.contracts.world import LatentWorldManifest
from training_data_collection.v72.families.base import WorldStateView
from training_data_collection.v72.generation.hidden_suite import (
    HiddenCaseRecord,
    HiddenSuiteManifest,
    MetamorphicTransformRecord,
    hidden_case_tree_sha256,
    validate_hidden_suite,
)

from .definition import PUBLIC_RULE_IDS, REWARD_COMPONENT_IDS, STAGE_IDS
from .render import (
    PRODUCTS_PATH,
    RAW_TRANSACTIONS_PATH,
    SCHEMA_PATH,
    apply_registered_corruptions,
    render_consistent_artifacts,
)


def _write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def _apply_hidden_change(case_root: Path, kind: str, rng: random.Random) -> None:
    paths = tuple(sorted((case_root / RAW_TRANSACTIONS_PATH).glob("transactions_*.csv")))
    for path in paths[:2]:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        keep = rows[max(1, len(rows) // 5) :]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(keep)
        path.write_text(stream.getvalue(), encoding="utf-8")
    if kind == "schema":
        path = case_root / SCHEMA_PATH
        schema = json.loads(path.read_bytes())
        schema["date_dimension"]["end"] = "2025-06-29"
        path.write_text(json.dumps(schema, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return
    if kind == "categorical":
        path = case_root / PRODUCTS_PATH
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        template = dict(rows[rng.randrange(len(rows))])
        template.update(
            {
                "product_id": "3999",
                "product_name": "Hidden Product",
                "category": "Hidden",
                "subcategory": "Qualification",
            }
        )
        rows.append(template)
        _write_tsv(path, fields, rows)
        return
    if kind != "scale":
        raise ValueError(f"unknown hidden warehouse change: {kind}")


def materialize_hidden_suite(
    *,
    world: LatentWorldManifest,
    private_root: Path,
    graph: ArtifactGraph,
    hidden_case_seed: int,
    relation_ids: tuple[str, ...],
    destination: Path,
) -> HiddenSuiteManifest:
    """Materialize three full public workspaces with independent semantic changes."""
    if len(relation_ids) != 6 or relation_ids != tuple(sorted(set(relation_ids))):
        raise ValueError("warehouse recovery hidden suite requires six exact relations")
    target = destination.resolve()
    cases = []
    view = WorldStateView(private_root=private_root, manifest=world)
    with atomic_directory(target, prefix="v72-warehouse-recovery-hidden-") as stage:
        for index, kind in enumerate(("schema", "categorical", "scale")):
            seed = derive_seed(hidden_case_seed, f"warehouse-recovery-hidden-{kind}")
            case_root = stage / f"case-{index:03d}"
            case_root.mkdir()
            render_consistent_artifacts(view, case_root)
            apply_registered_corruptions(world, graph, case_root)
            _apply_hidden_change(case_root, kind, random.Random(seed))
            file_paths = tuple(
                path.relative_to(stage).as_posix() for path in iter_regular_files(case_root)
            )
            cases.append(
                HiddenCaseRecord.identified(
                    case_name=case_root.name,
                    hidden_seed=seed,
                    file_paths=file_paths,
                    case_tree_sha256=hidden_case_tree_sha256(stage, file_paths),
                    public_rule_ids=PUBLIC_RULE_IDS,
                    relation_ids=relation_ids,
                    stage_ids=tuple(sorted(STAGE_IDS)),
                    reward_component_ids=REWARD_COMPONENT_IDS,
                )
            )
        records = tuple(sorted(cases, key=lambda item: item.case_id))
        transform = MetamorphicTransformRecord.identified(
            operator_id="warehouse-recovery.hidden-world-change.v1",
            source_case_id=records[0].case_id,
            transformed_case_id=records[1].case_id,
            expected_effect="semantic_change",
            covered_public_rule_ids=PUBLIC_RULE_IDS,
            covered_reward_component_ids=REWARD_COMPONENT_IDS,
        )
        manifest = HiddenSuiteManifest.identified(
            world_id=world.world_id,
            case_records=records,
            metamorphic_transform_records=(transform,),
            public_rule_coverage=PUBLIC_RULE_IDS,
            relation_coverage=relation_ids,
            stage_coverage=tuple(sorted(STAGE_IDS)),
            reward_component_coverage=REWARD_COMPONENT_IDS,
            hidden_tree_sha256=tree_sha256(stage),
            generator_implementation_digest=content_digest(
                {"implementation": "warehouse-recovery-hidden-workspaces", "version": 2}
            ),
        )
        validate_hidden_suite(stage, manifest)
    return manifest


__all__ = ["materialize_hidden_suite"]
