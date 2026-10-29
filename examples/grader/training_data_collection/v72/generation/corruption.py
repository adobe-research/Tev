"""Transactional execution boundary for registered corruption operators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from training_data_collection.common.filesystem import (
    atomic_directory,
    iter_regular_files,
    resolve_under,
    tree_manifest,
    tree_sha256,
)
from training_data_collection.v72.contracts.corruption import (
    CorruptionOperatorDeclaration,
)


class CorruptionExecutionError(ValueError):
    """Raised when a corruption violates its filesystem boundary."""


class MutableArtifactTree:
    """Safe relative-path access to one transactional artifact tree."""

    def __init__(self, root: Path) -> None:
        self.__root = root.resolve(strict=True)

    def read_bytes(self, relative: str) -> bytes:
        target = resolve_under(self.__root, relative)
        if target.is_symlink() or not target.is_file():
            raise CorruptionExecutionError(f"artifact is not a regular file: {relative}")
        return target.read_bytes()

    def write_bytes(self, relative: str, value: bytes) -> None:
        if type(value) is not bytes:
            raise CorruptionExecutionError("corruption writes must contain bytes")
        target = resolve_under(self.__root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)

    def unlink(self, relative: str) -> None:
        target = resolve_under(self.__root, relative)
        if target.is_symlink() or not target.is_file():
            raise CorruptionExecutionError(f"artifact is not a regular file: {relative}")
        target.unlink()


class CorruptionOperator(Protocol):
    """A registered deterministic mutation over a safe artifact tree."""

    declaration: CorruptionOperatorDeclaration

    def apply(self, tree: MutableArtifactTree, perturbation_seed: int) -> None: ...


@dataclass(frozen=True)
class CorruptionApplication:
    """Automatically measured filesystem effects from one operator execution."""

    baseline_tree_sha256: str
    corrupted_tree_sha256: str
    changed_paths: tuple[str, ...]


def _tree_digest_by_path(root: Path) -> dict[str, str]:
    return {str(record["path"]): str(record["sha256"]) for record in tree_manifest(root)}


def apply_corruption_transaction(
    *,
    baseline_root: Path,
    destination: Path,
    operator: CorruptionOperator,
    perturbation_seed: int,
) -> CorruptionApplication:
    """Copy a baseline, mutate its staging tree, and atomically publish the result."""
    if type(perturbation_seed) is not int or perturbation_seed < 0:
        raise CorruptionExecutionError("perturbation seed must be a nonnegative integer")
    baseline_root = baseline_root.resolve(strict=True)
    before_manifest = _tree_digest_by_path(baseline_root)
    before_tree = tree_sha256(baseline_root)

    with atomic_directory(destination, prefix="v72-corruption-") as stage:
        for source in iter_regular_files(baseline_root):
            relative = source.relative_to(baseline_root).as_posix()
            target = resolve_under(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

        operator.apply(MutableArtifactTree(stage), perturbation_seed)
        after_manifest = _tree_digest_by_path(stage)
        after_tree = tree_sha256(stage)
        changed_paths = tuple(
            sorted(
                path
                for path in set(before_manifest) | set(after_manifest)
                if before_manifest.get(path) != after_manifest.get(path)
            )
        )
        if not changed_paths or before_tree == after_tree:
            raise CorruptionExecutionError("corruption operator did not change the artifact tree")

    if tree_sha256(baseline_root) != before_tree:
        raise CorruptionExecutionError("corruption operator mutated the frozen baseline tree")
    if tree_sha256(destination) != after_tree:
        raise CorruptionExecutionError("published corruption differs from its staging tree")
    return CorruptionApplication(
        baseline_tree_sha256=before_tree,
        corrupted_tree_sha256=after_tree,
        changed_paths=changed_paths,
    )


__all__ = [
    "CorruptionApplication",
    "CorruptionExecutionError",
    "CorruptionOperator",
    "MutableArtifactTree",
    "apply_corruption_transaction",
]
