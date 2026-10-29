"""Canonical authority rendering and public-input freeze evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from training_data_collection.common.filesystem import (
    iter_regular_files,
    resolve_under,
    tree_sha256,
)
from training_data_collection.common.identity import canonical_json
from training_data_collection.v72.contracts.public_task import PublicTaskSpecification
from training_data_collection.v72.runtime.workspace import (
    PublicWorkspaceManifest,
    build_public_workspace,
    validate_public_workspace,
)


@dataclass(frozen=True)
class FrozenPublicInput:
    """Automatically measured public-input authority before private generation."""

    tree_sha256: str
    file_count: int

    @classmethod
    def capture(cls, input_root: Path) -> FrozenPublicInput:
        return cls(
            tree_sha256=tree_sha256(input_root),
            file_count=len(iter_regular_files(input_root)),
        )

    def require_unchanged(self, input_root: Path) -> None:
        if (
            tree_sha256(input_root) != self.tree_sha256
            or len(iter_regular_files(input_root)) != self.file_count
        ):
            raise ValueError("public input changed after its authority was frozen")


def render_public_authority(
    specification: PublicTaskSpecification,
    input_source: Path,
    destination: Path,
    *,
    software_source: Path | None = None,
) -> tuple[PublicWorkspaceManifest, FrozenPublicInput]:
    """Render the exact public workspace and capture its immutable input identity."""
    manifest = build_public_workspace(
        specification,
        input_source,
        destination,
        software_source=software_source,
    )
    validate_public_workspace(destination, specification, manifest)
    return manifest, FrozenPublicInput.capture(destination / "input")


def write_authority_model(root: Path, relative: str, model: BaseModel) -> Path:
    """Write one strict model as canonical JSON without overwriting authority."""
    destination = resolve_under(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(canonical_json(model.model_dump(mode="json")) + b"\n")
    return destination


__all__ = [
    "FrozenPublicInput",
    "render_public_authority",
    "write_authority_model",
]
