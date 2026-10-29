"""Source-derived family revision construction outside family role closures."""

from __future__ import annotations

import importlib
from pathlib import Path

from training_data_collection.common.identity import content_digest
from training_data_collection.v72.contracts.implementation import (
    ImplementationAuthority,
    ImplementationAuthorityKind,
)
from training_data_collection.v72.contracts.world import FamilyRevisionAuthority
from training_data_collection.v72.families.base import FamilyImplementation
from training_data_collection.v72.qualification.independence import (
    audit_implementation_independence,
    build_dependency_closure,
    reviewed_shared_module_allowlist,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_AUTHORITY_KINDS: tuple[ImplementationAuthorityKind, ...] = (
    "materializer",
    "reference",
    "alternative",
    "verifier",
)


def build_family_revision_authority(
    family: FamilyImplementation,
    logical_revision: str,
) -> FamilyRevisionAuthority:
    """Derive one family revision from its source and exact four role closures."""
    family_type = type(family)
    definition_module = family_type.__module__
    entrypoint_records = getattr(family, "authority_entrypoints", None)
    if entrypoint_records is None:
        if not definition_module.endswith(".definition"):
            raise ValueError(
                "family outside a registered definition module must declare role entrypoints"
            )
        package_module = definition_module.removesuffix(".definition")
        world_module = importlib.import_module(f"{package_module}.world")
        if getattr(world_module, "FAMILY_REVISION_ID", None) != logical_revision:
            raise ValueError("recipe logical revision differs from the registered family")
        qualification_module = importlib.import_module(f"{package_module}.qualification")
        entrypoint_records = getattr(qualification_module, "_ENTRYPOINTS", None)
    if type(entrypoint_records) is not tuple or any(
        type(record) is not tuple
        or len(record) != 3
        or type(record[1]) is not str
        or type(record[2]) is not str
        for record in entrypoint_records
    ):
        raise ValueError("family must register exact four-role implementation entrypoints")
    if tuple(record[0] for record in entrypoint_records) != _AUTHORITY_KINDS:
        raise ValueError("family must register exact four-role implementation entrypoints")
    closures = {
        kind: build_dependency_closure(entrypoint, _SOURCE_ROOT)
        for kind, entrypoint, _ in entrypoint_records
    }
    role_packages = {
        entrypoint.split(":", 1)[0].rsplit(".", 1)[0] for _, entrypoint, _ in entrypoint_records
    }
    if len(role_packages) != 1:
        raise ValueError("family role entrypoints must share one reviewed package")
    shared_module_ids = reviewed_shared_module_allowlist(role_packages.pop())
    authorities = {
        kind: ImplementationAuthority.identified(
            authority_kind=kind,
            source_tree_sha256=closures[kind].source_tree_sha256,
            entrypoint=entrypoint,
            dependency_closure_sha256=closures[kind].dependency_closure_sha256,
            allowed_shared_module_ids=shared_module_ids,
            algorithm_description_digest=content_digest({"algorithm_description": description}),
        )
        for kind, entrypoint, description in entrypoint_records
    }
    family_entrypoint = getattr(
        family,
        "family_implementation_entrypoint",
        f"{definition_module}:{family_type.__qualname__}.build_world",
    )
    if type(family_entrypoint) is not str:
        raise ValueError("family implementation entrypoint must be an importable callable")
    family_closure = build_dependency_closure(family_entrypoint, _SOURCE_ROOT)
    declared_digest = getattr(family, "implementation_digest", None)
    if type(declared_digest) is not str:
        raise ValueError("family implementation lacks its declared digest")
    return FamilyRevisionAuthority.identified(
        logical_revision=logical_revision,
        family_implementation_entrypoint=family_entrypoint,
        family_implementation_sha256=family_closure.dependency_closure_sha256,
        declared_implementation_sha256=declared_digest,
        materializer_authority=authorities["materializer"],
        reference_authority=authorities["reference"],
        alternative_authority=authorities["alternative"],
        verifier_authority=authorities["verifier"],
    )


def validate_family_revision_authority(authority: FamilyRevisionAuthority) -> None:
    """Rebuild a family authority from installed source and require exact equality."""
    family_closure = build_dependency_closure(
        authority.family_implementation_entrypoint,
        _SOURCE_ROOT,
    )
    if family_closure.dependency_closure_sha256 != authority.family_implementation_sha256:
        raise ValueError("family implementation closure differs from current source")
    module_name, qualname = authority.family_implementation_entrypoint.split(":", 1)
    target: object = importlib.import_module(module_name)
    for part in qualname.split(".")[:-1]:
        target = getattr(target, part)
    if getattr(target, "implementation_digest", None) != (authority.declared_implementation_sha256):
        raise ValueError("family declared implementation digest is stale")
    implementations = (
        authority.materializer_authority,
        authority.reference_authority,
        authority.alternative_authority,
        authority.verifier_authority,
    )
    for implementation in implementations:
        closure = build_dependency_closure(implementation.entrypoint, _SOURCE_ROOT)
        if (
            closure.source_tree_sha256 != implementation.source_tree_sha256
            or closure.dependency_closure_sha256 != implementation.dependency_closure_sha256
        ):
            raise ValueError(
                f"{implementation.authority_kind} authority differs from current source"
            )
    role_packages = {item.entrypoint.split(":", 1)[0].rsplit(".", 1)[0] for item in implementations}
    if len(role_packages) != 1:
        raise ValueError("family role authority packages differ")
    reviewed_shared_module_ids = reviewed_shared_module_allowlist(role_packages.pop())
    audit_implementation_independence(
        implementations,
        _SOURCE_ROOT,
        reviewed_shared_module_ids=reviewed_shared_module_ids,
    )


__all__ = [
    "build_family_revision_authority",
    "validate_family_revision_authority",
]
