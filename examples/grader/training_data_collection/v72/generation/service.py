"""Task-package validation and isolated policy-workspace reset service."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from training_data_collection.common.filesystem import (
    file_sha256,
    iter_regular_files,
    make_tree_writable,
    tree_manifest,
    tree_sha256,
)
from training_data_collection.common.identity import canonical_json, content_digest, content_id
from training_data_collection.v72.contracts.artifact import ArtifactGraph
from training_data_collection.v72.contracts.corruption import CorruptionEvent
from training_data_collection.v72.contracts.progress import ProgressContract
from training_data_collection.v72.contracts.prompt import PromptGenerationAuthority
from training_data_collection.v72.contracts.public_task import (
    FileManifest,
    ManifestEntry,
    PublicTaskSpecification,
    TaskPackageManifest,
)
from training_data_collection.v72.contracts.reward import RewardContract
from training_data_collection.v72.contracts.source import (
    GENERATED_SOURCE_PAYLOAD_PATH,
    SourcePayloadAuthority,
    SourceRecord,
    generated_source_payload_bytes,
)
from training_data_collection.v72.contracts.workflow import WorkflowGraph
from training_data_collection.v72.contracts.world import LatentWorldManifest, WorldRecipe
from training_data_collection.v72.families.base import GovernedSourceView, WorldStateView
from training_data_collection.v72.generation.hidden_suite import (
    HiddenSuiteManifest,
    validate_hidden_suite,
)
from training_data_collection.v72.generation.package import (
    PACKAGE_ROOT_PATHS,
    PRIVATE_AUTHORITY_PATHS,
    PRIVATE_ROOT_PATHS,
)
from training_data_collection.v72.generation.prompt import (
    build_prompt_generation_authority,
    require_authored_prompt_fidelity,
    require_no_private_prompt_leakage,
)
from training_data_collection.v72.qualification.artifact_graph import (
    validate_artifact_graph,
)
from training_data_collection.v72.qualification.family_revision import (
    validate_family_revision_authority,
)
from training_data_collection.v72.qualification.workflow import validate_workflow_graph
from training_data_collection.v72.runtime.workspace import (
    PublicWorkspaceManifest,
    build_public_workspace,
    validate_public_workspace,
)


class PackageValidationError(ValueError):
    """Raised when package bytes do not match their v7.2 authority records."""


@dataclass(frozen=True)
class ValidatedTaskPackage:
    root: Path
    task_manifest: TaskPackageManifest
    public_manifest: FileManifest
    private_manifest: FileManifest
    recipe: WorldRecipe
    world: LatentWorldManifest
    graph: ArtifactGraph
    workflow: WorkflowGraph
    corruptions: tuple[CorruptionEvent, ...]
    hidden_suite: HiddenSuiteManifest
    specification: PublicTaskSpecification
    prompt_generation_authority: PromptGenerationAuthority
    source_records: tuple[SourceRecord, ...]
    source_payload_authorities: tuple[SourcePayloadAuthority, ...]
    public_workspace_manifest: PublicWorkspaceManifest
    reward_contract: RewardContract | None
    progress_contract: ProgressContract | None


def _load_model(path: Path, model_type: type):
    try:
        return model_type.model_validate_json(path.read_bytes())
    except Exception as error:
        raise PackageValidationError(f"invalid authority model: {path.name}: {error}") from error


def _rebuild_file_manifest(root: Path, visibility: str) -> FileManifest:
    entries = tuple(
        ManifestEntry(
            path=str(record["path"]),
            visibility=visibility,
            sha256=str(record["sha256"]),
            size_bytes=int(record["size_bytes"]),
        )
        for record in tree_manifest(root)
    )
    return FileManifest.identified(
        visibility=visibility,
        tree_sha256=tree_sha256(root),
        entries=entries,
    )


def _load_model_sequence(path: Path, model_type: type) -> tuple:
    try:
        values = json.loads(path.read_bytes())
        if type(values) is not list:
            raise TypeError("authority sequence must be a JSON array")
        return tuple(model_type.model_validate_json(canonical_json(value)) for value in values)
    except Exception as error:
        raise PackageValidationError(f"invalid authority sequence: {path.name}: {error}") from error


def _corrupted_public_tree_sha256(public_root: Path) -> str:
    records = []
    for root_name in ("input", "software"):
        asset_root = public_root / root_name
        if not asset_root.is_dir():
            continue
        records.extend(
            {
                "path": f"{root_name}/{record['path']}",
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            for record in tree_manifest(asset_root)
            if not (root_name == "input" and record["path"] == "task_prompt.md")
        )
    return content_digest(records)


def validate_task_package(root: Path) -> ValidatedTaskPackage:
    """Load every authority record and recompute all package and tree identities."""
    if root.is_symlink() or not root.is_dir():
        raise PackageValidationError("task package root must be a real directory")
    root = root.resolve(strict=True)
    if tuple(sorted(path.name for path in root.iterdir())) != PACKAGE_ROOT_PATHS:
        raise PackageValidationError("task package root layout is not exact")
    public_root = root / "public"
    private_root = root / "private"
    authority_root = private_root / "authority"
    if tuple(sorted(path.name for path in private_root.iterdir())) != PRIVATE_ROOT_PATHS:
        raise PackageValidationError("private package root layout is not exact")
    actual_authority_paths = tuple(sorted(path.name for path in authority_root.iterdir()))
    required_authority_paths = set(PRIVATE_AUTHORITY_PATHS)
    optional_authority_paths = {"progress_contract.json", "reward_contract.json"}
    if not required_authority_paths.issubset(actual_authority_paths) or (
        set(actual_authority_paths) - required_authority_paths - optional_authority_paths
    ):
        raise PackageValidationError("private authority tree is incomplete")

    task_manifest = _load_model(root / "package_manifest.json", TaskPackageManifest)
    try:
        validate_family_revision_authority(task_manifest.family_revision)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise PackageValidationError(f"family revision authority is stale: {error}") from error
    public_manifest = _load_model(root / "public_manifest.json", FileManifest)
    private_manifest = _load_model(root / "private_manifest.json", FileManifest)
    if public_manifest != _rebuild_file_manifest(public_root, "public"):
        raise PackageValidationError("public file manifest differs from realized package bytes")
    if private_manifest != _rebuild_file_manifest(private_root, "private"):
        raise PackageValidationError("private file manifest differs from realized package bytes")

    recipe = _load_model(authority_root / "recipe.json", WorldRecipe)
    world = _load_model(authority_root / "world_manifest.json", LatentWorldManifest)
    graph = _load_model(authority_root / "artifact_graph.json", ArtifactGraph)
    workflow = _load_model(authority_root / "workflow.json", WorkflowGraph)
    hidden_suite = _load_model(authority_root / "hidden_suite_manifest.json", HiddenSuiteManifest)
    prompt_generation_authority = _load_model(
        authority_root / "prompt_generation.json", PromptGenerationAuthority
    )
    specification = _load_model(
        authority_root / "public_specification.json", PublicTaskSpecification
    )
    corruptions = _load_model_sequence(authority_root / "corruption_events.json", CorruptionEvent)
    source_records = _load_model_sequence(authority_root / "source_records.json", SourceRecord)
    source_payload_authorities = _load_model_sequence(
        authority_root / "source_payload_authorities.json", SourcePayloadAuthority
    )
    reward_contract = (
        _load_model(authority_root / "reward_contract.json", RewardContract)
        if (authority_root / "reward_contract.json").is_file()
        else None
    )
    progress_contract = (
        _load_model(authority_root / "progress_contract.json", ProgressContract)
        if (authority_root / "progress_contract.json").is_file()
        else None
    )
    if (reward_contract is None) != (task_manifest.reward_contract_id is None) or (
        progress_contract is None
    ) != (task_manifest.progress_contract_id is None):
        raise PackageValidationError("dense reward authority files differ from task manifest")
    if (
        reward_contract is not None
        and reward_contract.contract_id != task_manifest.reward_contract_id
    ):
        raise PackageValidationError("reward contract identity differs from task manifest")
    if progress_contract is not None:
        if reward_contract is None:
            raise PackageValidationError("progress authority lacks final reward authority")
        if (
            progress_contract.progress_contract_id != task_manifest.progress_contract_id
            or progress_contract.workflow_id != workflow.workflow_id
            or progress_contract.family_revision_id != recipe.family_revision_id
            or progress_contract.final_reward_contract_id != reward_contract.contract_id
        ):
            raise PackageValidationError("progress contract has stale package bindings")

    if not corruptions or not source_records:
        raise PackageValidationError("package corruption and source authority cannot be empty")
    if world.recipe_id != recipe.recipe_id or world.family_revision_id != recipe.family_revision_id:
        raise PackageValidationError("world authority does not bind the selected recipe")
    WorldStateView(private_root, world)
    if graph.world_id != world.world_id or workflow.artifact_graph_id != graph.graph_id:
        raise PackageValidationError("graph and workflow do not bind the selected world")
    if workflow.curriculum_layer != recipe.curriculum_layer:
        raise PackageValidationError("workflow curriculum layer differs from recipe")
    validate_artifact_graph(graph)
    validate_workflow_graph(workflow, graph)
    if hidden_suite.world_id != world.world_id:
        raise PackageValidationError("hidden suite does not bind the selected world")
    validate_hidden_suite(private_root / "hidden_suite", hidden_suite)
    if specification.authored_prompt is not None:
        require_authored_prompt_fidelity(specification, specification.authored_prompt.text)
        require_no_private_prompt_leakage(
            specification,
            specification.authored_prompt.text,
            private_roots=(
                private_root / "world",
                private_root / "hidden_suite",
                private_root / "governed_sources",
            ),
            public_roots=tuple(
                path for path in (public_root / "input", public_root / "software") if path.is_dir()
            ),
        )
    expected_prompt_generation_authority = build_prompt_generation_authority(
        specification,
        transcript=prompt_generation_authority.transcript,
    )
    if prompt_generation_authority != expected_prompt_generation_authority:
        raise PackageValidationError("prompt generation authority differs from prompt bytes")
    if tuple(sorted(record.source_id for record in source_records)) != recipe.source_record_ids:
        raise PackageValidationError("source records do not exactly bind the recipe")
    source_records_by_id = {record.source_id: record for record in source_records}
    for authority in source_payload_authorities:
        record = source_records_by_id.get(authority.source_id)
        expected_realization = (
            task_manifest.family_revision.materializer_authority.authority_id
            if authority.realization_kind == "deterministic_generator"
            else record.acquisition_record_id
            if record is not None
            else None
        )
        if authority.realization_authority_id != expected_realization:
            raise PackageValidationError(
                "source payload realization differs from exact producer authority"
            )
        if record is not None and record.source_kind == "generated":
            source_root = private_root / "governed_sources" / record.source_id.rsplit(":", 1)[1]
            source_files = tuple(iter_regular_files(source_root))
            if (
                len(source_files) != 1
                or source_files[0].relative_to(source_root).as_posix()
                != GENERATED_SOURCE_PAYLOAD_PATH
                or source_files[0].read_bytes()
                != generated_source_payload_bytes(
                    origin_identity=record.origin_identity,
                    immutable_version=record.immutable_version,
                )
            ):
                raise PackageValidationError(
                    "generated source payload differs from deterministic authority"
                )
    try:
        GovernedSourceView(
            private_root=private_root,
            source_records=source_records,
            payload_authorities=source_payload_authorities,
        )
    except ValueError as error:
        raise PackageValidationError(f"governed source payload is stale: {error}") from error
    if any(event.world_id != world.world_id for event in corruptions):
        raise PackageValidationError("corruption events do not bind the selected world")
    if any(
        event.baseline_tree_sha256 != corruptions[index - 1].corrupted_tree_sha256
        for index, event in enumerate(corruptions)
        if index
    ):
        raise PackageValidationError("corruption event tree identities are discontinuous")
    if corruptions[-1].corrupted_tree_sha256 != _corrupted_public_tree_sha256(public_root):
        raise PackageValidationError("corruption evidence differs from packaged public input")

    task_specification_sha256 = None
    workspace_paths = ("prompt.md", "input", "output", "software")
    software_tree_sha256 = None
    software_file_count = 0
    if specification.environment.workspace_layout == "machine_contract":
        loaded_public_spec = PublicTaskSpecification.model_validate_json(
            (public_root / "task_spec.json").read_bytes()
        )
        if loaded_public_spec != specification:
            raise PackageValidationError("public and private task specifications differ")
        task_specification_sha256 = file_sha256(public_root / "task_spec.json")
        workspace_paths = ("task_spec.json", "prompt.md", "input", "output")
    else:
        software_tree_sha256 = tree_sha256(public_root / "software")
        software_file_count = len(iter_regular_files(public_root / "software"))
    public_workspace = PublicWorkspaceManifest.identified(
        specification_id=specification.specification_id,
        task_specification_sha256=task_specification_sha256,
        prompt_sha256=file_sha256(public_root / "prompt.md"),
        input_tree_sha256=tree_sha256(public_root / "input"),
        input_file_count=len(iter_regular_files(public_root / "input")),
        software_tree_sha256=software_tree_sha256,
        software_file_count=software_file_count,
        workspace_paths=workspace_paths,
    )
    validate_public_workspace(public_root, specification, public_workspace)
    if iter_regular_files(public_root / "output"):
        raise PackageValidationError("packaged public output must be empty")

    corruption_manifest_id = content_id(
        "v72-corruption-manifest",
        [event.model_dump(mode="json") for event in corruptions],
    )
    policy_digest_set_id = content_id(
        "v72-policy-digest-set",
        [binding.model_dump(mode="json") for binding in recipe.policy_digest_set],
    )
    expected_bindings = {
        "family_revision": task_manifest.family_revision,
        "recipe_id": recipe.recipe_id,
        "world_id": world.world_id,
        "artifact_graph_id": graph.graph_id,
        "workflow_id": workflow.workflow_id,
        "corruption_manifest_id": corruption_manifest_id,
        "hidden_suite_id": hidden_suite.hidden_suite_id,
        "public_specification_id": specification.specification_id,
        "prompt_sha256": file_sha256(public_root / "prompt.md"),
        "public_manifest_id": public_manifest.manifest_id,
        "private_manifest_id": private_manifest.manifest_id,
        "public_tree_sha256": public_manifest.tree_sha256,
        "private_tree_sha256": private_manifest.tree_sha256,
        "public_input_tree_sha256": public_workspace.input_tree_sha256,
        "scale_id": recipe.scale_id,
        "curriculum_layer": recipe.curriculum_layer,
        "behavior_seed": recipe.behavior_seed,
        "source_record_ids": recipe.source_record_ids,
        "policy_digest_set_id": policy_digest_set_id,
        "reward_contract_id": (
            reward_contract.contract_id if reward_contract is not None else None
        ),
        "progress_contract_id": (
            progress_contract.progress_contract_id if progress_contract is not None else None
        ),
    }
    if task_manifest.family_revision.logical_revision != recipe.family_revision_id:
        raise PackageValidationError(
            "task family revision does not bind the recipe logical revision"
        )
    if any(getattr(task_manifest, field) != value for field, value in expected_bindings.items()):
        raise PackageValidationError("task package manifest has stale authority bindings")
    if task_manifest.task_id != task_manifest.expected_task_id():
        raise PackageValidationError("task package identity is stale")

    return ValidatedTaskPackage(
        root=root,
        task_manifest=task_manifest,
        public_manifest=public_manifest,
        private_manifest=private_manifest,
        recipe=recipe,
        world=world,
        graph=graph,
        workflow=workflow,
        corruptions=corruptions,
        hidden_suite=hidden_suite,
        specification=specification,
        prompt_generation_authority=prompt_generation_authority,
        source_records=source_records,
        source_payload_authorities=source_payload_authorities,
        public_workspace_manifest=public_workspace,
        reward_contract=reward_contract,
        progress_contract=progress_contract,
    )


def create_policy_workspace(package_root: Path, destination: Path) -> PublicWorkspaceManifest:
    """Copy only frozen public authority into a new empty policy workspace."""
    package = validate_task_package(package_root)
    return build_public_workspace(
        package.specification,
        package.root / "public" / "input",
        destination,
        software_source=(
            package.root / "public" / "software"
            if (package.root / "public" / "software").is_dir()
            else None
        ),
    )


def reset_policy_workspace(package_root: Path, destination: Path) -> PublicWorkspaceManifest:
    """Reset a validated policy workspace to the package's frozen public state."""
    package = validate_task_package(package_root)
    if destination.exists():
        validate_public_workspace(
            destination,
            package.specification,
            package.public_workspace_manifest,
        )
        make_tree_writable(destination)
        shutil.rmtree(destination)
    return build_public_workspace(
        package.specification,
        package.root / "public" / "input",
        destination,
        software_source=(
            package.root / "public" / "software"
            if (package.root / "public" / "software").is_dir()
            else None
        ),
    )


__all__ = [
    "PackageValidationError",
    "ValidatedTaskPackage",
    "create_policy_workspace",
    "reset_policy_workspace",
    "validate_task_package",
]
