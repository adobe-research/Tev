"""Deterministic public workspace construction and isolation checks."""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import ValidationInfo, model_validator

from training_data_collection.common.filesystem import (
    atomic_directory,
    file_sha256,
    iter_regular_files,
    make_tree_read_only,
    resolve_under,
    tree_sha256,
)
from training_data_collection.common.identity import content_id
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)
from training_data_collection.v72.contracts.public_task import (
    PublicInputRole,
    PublicTaskSpecification,
)
from training_data_collection.v72.generation.prompt import (
    render_canonical_prompt,
    require_canonical_prompt,
)

PUBLIC_WORKSPACE_PATHS = ("task_spec.json", "prompt.md", "input", "output")
BENCHMARK_NATIVE_WORKSPACE_PATHS = ("prompt.md", "input", "output", "software")


class PublicWorkspaceManifest(StrictFrozenModel):
    """Content identity of the immutable policy-visible workspace inputs."""

    schema_version: Literal["v7.2.public-workspace.v1"] = "v7.2.public-workspace.v1"
    workspace_id: ContentIdentifier
    specification_id: ContentIdentifier
    task_specification_sha256: Sha256Digest | None
    prompt_sha256: Sha256Digest
    input_tree_sha256: Sha256Digest
    input_file_count: int
    software_tree_sha256: Sha256Digest | None = None
    software_file_count: int = 0
    workspace_paths: tuple[str, ...] = PUBLIC_WORKSPACE_PATHS

    @model_validator(mode="after")
    def validate_manifest(self, info: ValidationInfo) -> Self:
        if self.input_file_count < 0 or self.software_file_count < 0:
            raise ValueError("public workspace file counts cannot be negative")
        expected_paths = (
            PUBLIC_WORKSPACE_PATHS
            if self.task_specification_sha256 is not None
            else BENCHMARK_NATIVE_WORKSPACE_PATHS
        )
        if self.workspace_paths != expected_paths:
            raise ValueError("public workspace paths differ from the selected layout")
        if (self.software_tree_sha256 is None) != (self.software_file_count == 0):
            raise ValueError("software tree digest and file count must be present together")
        if self.task_specification_sha256 is None and self.software_tree_sha256 is None:
            raise ValueError("benchmark-native workspaces require staged software")
        if not (info.context or {}).get("skip_identity") and (
            self.workspace_id != self.expected_workspace_id()
        ):
            raise ValueError("workspace_id does not match public workspace authority")
        return self

    def expected_workspace_id(self) -> str:
        return content_id(
            "v72-public-workspace",
            self.model_dump(mode="json", exclude={"workspace_id"}),
        )

    @classmethod
    def identified(
        cls,
        *,
        specification_id: str,
        task_specification_sha256: str,
        prompt_sha256: str,
        input_tree_sha256: str,
        input_file_count: int,
        software_tree_sha256: str | None = None,
        software_file_count: int = 0,
        workspace_paths: tuple[str, ...] = PUBLIC_WORKSPACE_PATHS,
    ) -> Self:
        values = {
            "specification_id": specification_id,
            "task_specification_sha256": task_specification_sha256,
            "prompt_sha256": prompt_sha256,
            "input_tree_sha256": input_tree_sha256,
            "input_file_count": input_file_count,
            "software_tree_sha256": software_tree_sha256,
            "software_file_count": software_file_count,
            "workspace_paths": workspace_paths,
        }
        provisional = cls.model_validate(
            {"workspace_id": f"v72-public-workspace:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(workspace_id=provisional.expected_workspace_id(), **values)


def _role_matches_file(role: PublicInputRole, relative_path: str) -> bool:
    role_path = PurePosixPath(role.path)
    candidate = PurePosixPath(relative_path)
    return (
        candidate == role_path if role.path_kind == "file" else candidate.is_relative_to(role_path)
    )


def _validate_asset_tree(
    asset_root: Path,
    specification: PublicTaskSpecification,
    *,
    role_root: Literal["input", "software"],
) -> None:
    files = iter_regular_files(asset_root)
    relative_files = tuple(path.relative_to(asset_root).as_posix() for path in files)
    roles = tuple(role for role in specification.public_inputs if role.root == role_root)
    for role in roles:
        target = resolve_under(asset_root, role.path)
        if role.required and not target.exists():
            raise ValueError(f"required public input role is missing: {role.role_id}")
        if target.exists() and target.is_symlink():
            raise ValueError(f"public input roles cannot resolve to symlinks: {role.role_id}")
        if target.exists() and role.path_kind == "file" and not target.is_file():
            raise ValueError(f"public input role must resolve to a file: {role.role_id}")
        if target.exists() and role.path_kind == "directory" and not target.is_dir():
            raise ValueError(f"public input role must resolve to a directory: {role.role_id}")
    for relative_path in relative_files:
        if (
            role_root == "input"
            and specification.environment.workspace_layout == "benchmark_native"
            and relative_path == "task_prompt.md"
        ):
            continue
        matches = [role.role_id for role in roles if _role_matches_file(role, relative_path)]
        if len(matches) != 1:
            raise ValueError(
                f"public input file must be covered by exactly one role: {relative_path!r}"
            )

    role_paths = [PurePosixPath(role.path) for role in roles]
    for directory in sorted(path for path in asset_root.rglob("*") if path.is_dir()):
        relative = PurePosixPath(directory.relative_to(asset_root).as_posix())
        declared = any(
            relative == role_path
            or relative.is_relative_to(role_path)
            or role_path.is_relative_to(relative)
            for role_path in role_paths
        )
        if not declared and not (
            role_root == "input"
            and specification.environment.workspace_layout == "benchmark_native"
            and relative == PurePosixPath(".")
        ):
            raise ValueError(f"undeclared public input directory: {relative.as_posix()!r}")


def _task_specification_bytes(specification: PublicTaskSpecification) -> bytes:
    payload = json.dumps(
        specification.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (payload + "\n").encode("utf-8")


def build_public_workspace(
    specification: PublicTaskSpecification,
    input_source: Path,
    destination: Path,
    *,
    software_source: Path | None = None,
) -> PublicWorkspaceManifest:
    """Publish an exact policy workspace after freezing all public authority inputs."""
    if input_source.is_symlink():
        raise ValueError("public input source cannot be a symlink")
    source = input_source.resolve(strict=True)
    target = destination.resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError("public workspace destination cannot be inside its input source")
    _validate_asset_tree(source, specification, role_root="input")
    native_layout = specification.environment.workspace_layout == "benchmark_native"
    resolved_software: Path | None = None
    if native_layout:
        if software_source is None or software_source.is_symlink():
            raise ValueError("benchmark-native workspace requires a real software source")
        resolved_software = software_source.resolve(strict=True)
        _validate_asset_tree(resolved_software, specification, role_root="software")
    elif software_source is not None:
        raise ValueError("machine-contract workspace cannot stage supplemental software")

    with atomic_directory(target, prefix="v72-public-workspace-") as stage:
        input_target = stage / "input"
        output_target = stage / "output"
        shutil.copytree(source, input_target)
        output_target.mkdir()
        prompt = render_canonical_prompt(specification)
        (stage / "prompt.md").write_text(prompt, encoding="utf-8")
        task_specification_sha256 = None
        software_tree_sha256 = None
        software_file_count = 0
        workspace_paths = PUBLIC_WORKSPACE_PATHS
        if native_layout:
            task_prompt_target = input_target / "task_prompt.md"
            if task_prompt_target.exists():
                require_canonical_prompt(
                    specification,
                    task_prompt_target.read_text(encoding="utf-8"),
                )
            else:
                task_prompt_target.write_text(prompt, encoding="utf-8")
            assert resolved_software is not None
            software_target = stage / "software"
            shutil.copytree(resolved_software, software_target)
            software_tree_sha256 = tree_sha256(software_target)
            software_file_count = len(iter_regular_files(software_target))
            workspace_paths = BENCHMARK_NATIVE_WORKSPACE_PATHS
            make_tree_read_only(software_target)
        else:
            (stage / "task_spec.json").write_bytes(_task_specification_bytes(specification))
            task_specification_sha256 = file_sha256(stage / "task_spec.json")
        manifest = PublicWorkspaceManifest.identified(
            specification_id=specification.specification_id,
            task_specification_sha256=task_specification_sha256,
            prompt_sha256=file_sha256(stage / "prompt.md"),
            input_tree_sha256=tree_sha256(input_target),
            input_file_count=len(iter_regular_files(input_target)),
            software_tree_sha256=software_tree_sha256,
            software_file_count=software_file_count,
            workspace_paths=workspace_paths,
        )
        make_tree_read_only(input_target)
        if not native_layout:
            (stage / "task_spec.json").chmod(0o444)
        (stage / "prompt.md").chmod(0o444)
        validate_public_workspace(stage, specification, manifest)
        if iter_regular_files(output_target):
            raise ValueError("new public workspace output must be empty")
    return manifest


def validate_public_workspace(
    root: Path,
    specification: PublicTaskSpecification,
    manifest: PublicWorkspaceManifest,
) -> None:
    """Verify exact layout, content authority, read-only inputs, and private isolation."""
    if root.is_symlink():
        raise ValueError("public workspace root must be a real directory")
    workspace = root.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("public workspace root must be a real directory")
    actual_paths = tuple(sorted(path.name for path in workspace.iterdir()))
    if actual_paths != tuple(sorted(manifest.workspace_paths)):
        raise ValueError("public workspace paths differ from their exact manifest")

    task_spec_path = workspace / "task_spec.json"
    prompt_path = workspace / "prompt.md"
    input_root = workspace / "input"
    output_root = workspace / "output"
    if task_spec_path.is_symlink() or prompt_path.is_symlink():
        raise ValueError("public task authority files cannot be symlinks")
    native_layout = specification.environment.workspace_layout == "benchmark_native"
    if native_layout:
        if task_spec_path.exists():
            raise ValueError("benchmark-native workspace cannot expose task_spec.json")
        task_prompt_path = workspace / "input/task_prompt.md"
        if task_prompt_path.is_symlink() or not task_prompt_path.is_file():
            raise ValueError("benchmark-native workspace lacks input/task_prompt.md")
        require_canonical_prompt(specification, task_prompt_path.read_text(encoding="utf-8"))
        if task_prompt_path.read_bytes() != prompt_path.read_bytes():
            raise ValueError("benchmark-native prompt mirror differs from task prompt")
    else:
        loaded = PublicTaskSpecification.model_validate_json(task_spec_path.read_bytes())
        if loaded != specification:
            raise ValueError("task_spec.json differs from selected public task authority")
    require_canonical_prompt(specification, prompt_path.read_text(encoding="utf-8"))
    _validate_asset_tree(input_root, specification, role_root="input")
    software_root = workspace / "software"
    if native_layout:
        _validate_asset_tree(software_root, specification, role_root="software")
    iter_regular_files(output_root)

    if not native_layout and file_sha256(task_spec_path) != manifest.task_specification_sha256:
        raise ValueError("task_spec.json digest differs from workspace authority")
    if file_sha256(prompt_path) != manifest.prompt_sha256:
        raise ValueError("prompt.md digest differs from workspace authority")
    if tree_sha256(input_root) != manifest.input_tree_sha256:
        raise ValueError("public input tree digest differs from workspace authority")
    if len(iter_regular_files(input_root)) != manifest.input_file_count:
        raise ValueError("public input file count differs from workspace authority")
    if native_layout and (
        tree_sha256(software_root) != manifest.software_tree_sha256
        or len(iter_regular_files(software_root)) != manifest.software_file_count
    ):
        raise ValueError("public software tree differs from workspace authority")
    if specification.specification_id != manifest.specification_id:
        raise ValueError("workspace specification identity differs from selected authority")
    if manifest.workspace_id != manifest.expected_workspace_id():
        raise ValueError("workspace manifest content identity is stale")

    immutable_paths = (prompt_path, input_root, *input_root.rglob("*"))
    if native_layout:
        immutable_paths = (*immutable_paths, software_root, *software_root.rglob("*"))
    else:
        immutable_paths = (task_spec_path, *immutable_paths)
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if any(path.stat().st_mode & write_bits for path in immutable_paths):
        raise ValueError("public task specification, prompt, and input tree must be read-only")
