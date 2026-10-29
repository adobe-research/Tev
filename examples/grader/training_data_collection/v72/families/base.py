"""Cohesive family protocol for world-first v7.2 task construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from training_data_collection.common.filesystem import (
    iter_regular_files,
    resolve_under,
    tree_sha256,
)
from training_data_collection.common.identity import canonical_json
from training_data_collection.v72.contracts.artifact import ArtifactGraph
from training_data_collection.v72.contracts.corruption import CorruptionEvent
from training_data_collection.v72.contracts.public_task import PublicTaskSpecification
from training_data_collection.v72.contracts.source import SourcePayloadAuthority, SourceRecord
from training_data_collection.v72.contracts.workflow import WorkflowGraph
from training_data_collection.v72.contracts.world import LatentWorldManifest, WorldRecipe
from training_data_collection.v72.generation.hidden_suite import HiddenSuiteManifest


def canonical_source_payload_bytes(root: Path) -> bytes:
    """Encode every regular source payload file with its canonical relative path."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("governed source payload root must be a real directory")
    files = iter_regular_files(root)
    if not files:
        raise ValueError("governed source payload must contain a regular file")
    return canonical_json(
        [
            {
                "path": path.relative_to(root).as_posix(),
                "content_hex": path.read_bytes().hex(),
            }
            for path in files
        ]
    )


@dataclass(frozen=True)
class WorldStateView:
    """Verified read access to the fixed private world-state subtree."""

    private_root: Path
    manifest: LatentWorldManifest

    def __post_init__(self) -> None:
        world_root = self.private_root / "world"
        if tree_sha256(world_root) != self.manifest.state_tree_sha256:
            raise ValueError("private world state differs from its manifest")

    @property
    def world_root(self) -> Path:
        return self.private_root / "world"

    def entity_path(self, entity_id: str) -> Path:
        matches = [
            entity.state_path
            for entity in self.manifest.entity_index
            if entity.entity_id == entity_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown world entity: {entity_id}")
        path = resolve_under(self.world_root, matches[0])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"world entity is not a regular file: {entity_id}")
        return path

    def read_entity_bytes(self, entity_id: str) -> bytes:
        return self.entity_path(entity_id).read_bytes()


@dataclass(frozen=True)
class GovernedSourceView:
    """Read-only, content-verified access to exact governed source payload bytes."""

    private_root: Path
    source_records: tuple[SourceRecord, ...]
    payload_authorities: tuple[SourcePayloadAuthority, ...]

    def __post_init__(self) -> None:
        record_by_id = {item.source_id: item for item in self.source_records}
        authority_by_id = {item.source_id: item for item in self.payload_authorities}
        if (
            len(record_by_id) != len(self.source_records)
            or len(authority_by_id) != len(self.payload_authorities)
            or set(record_by_id) != set(authority_by_id)
        ):
            raise ValueError("governed source records and payload authorities are not exact")
        root = self.private_root / "governed_sources"
        expected_names = {source_id.rsplit(":", 1)[1] for source_id in record_by_id}
        if (
            root.is_symlink()
            or not root.is_dir()
            or {path.name for path in root.iterdir()} != expected_names
        ):
            raise ValueError("governed source root layout is incomplete")
        for source_id, record in record_by_id.items():
            authority = authority_by_id[source_id]
            payload_root = root / source_id.rsplit(":", 1)[1]
            if (
                payload_root.is_symlink()
                or not payload_root.is_dir()
                or not iter_regular_files(payload_root)
                or authority.source_tree_sha256 != record.source_tree_sha256
                or authority.acquisition_record_id != record.acquisition_record_id
                or tree_sha256(payload_root) != record.source_tree_sha256
            ):
                raise ValueError("governed source payload differs from its source authority")

    def source_root(self, source_id: str) -> Path:
        if source_id not in {item.source_id for item in self.source_records}:
            raise KeyError(f"unknown governed source: {source_id}")
        return self.private_root / "governed_sources" / source_id.rsplit(":", 1)[1]

    def read_source_bytes(self, source_id: str) -> bytes:
        return canonical_source_payload_bytes(self.source_root(source_id))


class FamilyImplementation(Protocol):
    """Section 8 family operations executed by the generic package builder."""

    implementation_digest: str

    def build_world(
        self,
        recipe: WorldRecipe,
        private_root: Path,
        governed_sources: GovernedSourceView,
    ) -> LatentWorldManifest: ...

    def build_workflow(self, world: LatentWorldManifest) -> WorkflowGraph: ...

    def render_artifacts(
        self,
        world: LatentWorldManifest,
        workflow: WorkflowGraph,
        public_root: Path,
        private_root: Path,
    ) -> ArtifactGraph: ...

    def apply_corruptions(
        self,
        world: LatentWorldManifest,
        graph: ArtifactGraph,
        public_root: Path,
    ) -> tuple[ArtifactGraph, tuple[CorruptionEvent, ...]]: ...

    def build_public_specification(
        self,
        graph: ArtifactGraph,
        workflow: WorkflowGraph,
    ) -> PublicTaskSpecification: ...

    def build_hidden_suite(
        self,
        recipe: WorldRecipe,
        world: LatentWorldManifest,
        private_root: Path,
    ) -> HiddenSuiteManifest: ...


__all__ = [
    "FamilyImplementation",
    "GovernedSourceView",
    "WorldStateView",
    "canonical_source_payload_bytes",
]
