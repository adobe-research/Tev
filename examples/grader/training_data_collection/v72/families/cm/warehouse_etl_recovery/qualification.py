"""Four-role source independence authority for warehouse ETL recovery."""

from __future__ import annotations

from pathlib import Path

from training_data_collection.v72.contracts.implementation import ImplementationAuthorityKind
from training_data_collection.v72.qualification.independence import (
    ImplementationIndependenceEvidence,
    audit_implementation_independence,
    build_implementation_authority,
    reviewed_shared_module_allowlist,
)

_PACKAGE = "training_data_collection.v72.families.cm.warehouse_etl_recovery"
_ENTRYPOINTS: tuple[tuple[ImplementationAuthorityKind, str, str], ...] = (
    (
        "materializer",
        f"{_PACKAGE}.world:run_materializer_process",
        "Generate fresh relational truth and defect plans from recipe seed namespaces.",
    ),
    (
        "reference",
        f"{_PACKAGE}.reference:run_reference_process",
        "Normalize public mixed-format sources and directly load a canonical star schema.",
    ),
    (
        "alternative",
        f"{_PACKAGE}.alternative:run_alternative_process",
        "Independently reconcile public partitions and build the warehouse with separate code.",
    ),
    (
        "verifier",
        f"{_PACKAGE}.verifier:run_verifier_process",
        "Recompute expectations from public inputs and inspect sealed SQLite semantics.",
    ),
)
_REVIEWED_SHARED_MODULE_IDS = reviewed_shared_module_allowlist(_PACKAGE)


def audit_role_independence(source_root: Path) -> ImplementationIndependenceEvidence:
    """Execute dependency-closure independence checks for all four roles."""
    authorities = tuple(
        build_implementation_authority(
            authority_kind=kind,
            entrypoint=entrypoint,
            source_root=source_root,
            allowed_shared_module_ids=_REVIEWED_SHARED_MODULE_IDS,
            algorithm_description=description,
        )
        for kind, entrypoint, description in _ENTRYPOINTS
    )
    return audit_implementation_independence(
        authorities,
        source_root,
        reviewed_shared_module_ids=_REVIEWED_SHARED_MODULE_IDS,
    )


__all__ = ["audit_role_independence"]
