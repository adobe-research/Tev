"""Transactional world-to-package orchestration for v7.2 families."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from training_data_collection.common.filesystem import (
    atomic_directory,
    file_sha256,
    iter_regular_files,
    make_tree_read_only,
    make_tree_writable,
    resolve_under,
    tree_manifest,
    tree_sha256,
)
from training_data_collection.common.identity import canonical_json, content_id
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
from training_data_collection.v72.families.base import (
    FamilyImplementation,
    GovernedSourceView,
    WorldStateView,
)
from training_data_collection.v72.generation.hidden_suite import (
    HiddenSuiteManifest,
    validate_hidden_suite,
)
from training_data_collection.v72.generation.prompt import (
    OfflinePromptAuthoringInput,
    build_prompt_generation_authority,
    freeze_offline_authored_prompt,
)
from training_data_collection.v72.generation.render import (
    FrozenPublicInput,
    render_public_authority,
    write_authority_model,
)
from training_data_collection.v72.qualification.artifact_graph import (
    validate_artifact_graph,
)
from training_data_collection.v72.qualification.family_revision import (
    build_family_revision_authority,
)
from training_data_collection.v72.qualification.workflow import validate_workflow_graph
from training_data_collection.v72.runtime.workspace import PublicWorkspaceManifest

PACKAGE_ROOT_PATHS = (
    "package_manifest.json",
    "private",
    "private_manifest.json",
    "public",
    "public_manifest.json",
)
PRIVATE_ROOT_PATHS = ("authority", "governed_sources", "hidden_suite", "world")
PRIVATE_AUTHORITY_PATHS = (
    "artifact_graph.json",
    "corruption_events.json",
    "hidden_suite_manifest.json",
    "prompt_generation.json",
    "public_specification.json",
    "recipe.json",
    "source_payload_authorities.json",
    "source_records.json",
    "workflow.json",
    "world_manifest.json",
)
PRIVATE_OPTIONAL_AUTHORITY_PATHS = ("progress_contract.json", "reward_contract.json")


class PackageBuildError(ValueError):
    """Raised when a family cannot produce a complete authoritative package."""


@dataclass(frozen=True)
class SourcePayloadInput:
    """Host input root paired with exact source-payload producer authority."""

    authority: SourcePayloadAuthority
    root: Path


def create_generated_source_payload_input(
    source_record: SourceRecord,
    destination: Path,
    *,
    materializer_authority_id: str,
) -> SourcePayloadInput:
    """Materialize and bind the deterministic producer payload for a generated source."""
    if source_record.source_kind != "generated":
        raise ValueError("only generated records can use deterministic payload materialization")
    payload = generated_source_payload_bytes(
        origin_identity=source_record.origin_identity,
        immutable_version=source_record.immutable_version,
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / GENERATED_SOURCE_PAYLOAD_PATH).write_bytes(payload)
    if tree_sha256(destination) != source_record.source_tree_sha256:
        raise ValueError("generated payload bytes differ from the source record tree")
    return SourcePayloadInput(
        authority=SourcePayloadAuthority.identified(
            source_record=source_record,
            realization_kind="deterministic_generator",
            realization_authority_id=materializer_authority_id,
        ),
        root=destination,
    )


def stage_generated_source_view(
    source_records: tuple[SourceRecord, ...],
    private_root: Path,
    *,
    materializer_authority_id: str | None = None,
) -> GovernedSourceView:
    """Stage exact generated-source bytes for direct family qualification execution."""
    private_root.mkdir(parents=True, exist_ok=True)
    governed_root = private_root / "governed_sources"
    governed_root.mkdir()
    if source_records and materializer_authority_id is None:
        raise ValueError("generated source staging requires exact materializer authority")
    inputs = tuple(
        create_generated_source_payload_input(
            record,
            governed_root / record.source_id.rsplit(":", 1)[1],
            materializer_authority_id=materializer_authority_id or "",
        )
        for record in source_records
    )
    make_tree_read_only(governed_root)
    return GovernedSourceView(
        private_root=private_root,
        source_records=source_records,
        payload_authorities=tuple(item.authority for item in inputs),
    )


@dataclass(frozen=True)
class PackageBuildResult:
    """Validated identities and final path for one atomically published package."""

    package_root: Path
    task_manifest: TaskPackageManifest
    public_file_manifest: FileManifest
    private_file_manifest: FileManifest
    public_workspace_manifest: PublicWorkspaceManifest
    frozen_public_input: FrozenPublicInput
    world_manifest: LatentWorldManifest
    artifact_graph: ArtifactGraph
    workflow: WorkflowGraph
    corruptions: tuple[CorruptionEvent, ...]
    hidden_suite: HiddenSuiteManifest
    public_specification: PublicTaskSpecification
    prompt_generation_authority: PromptGenerationAuthority
    source_records: tuple[SourceRecord, ...]
    source_payload_authorities: tuple[SourcePayloadAuthority, ...]
    reward_contract: RewardContract | None
    progress_contract: ProgressContract | None


def _require_model_identity(model: object, identity_name: str, expected: str) -> None:
    actual = getattr(model, identity_name, None)
    if actual != expected:
        raise PackageBuildError(f"stale {identity_name} in family build result")


def _file_manifest(root: Path, visibility: str) -> FileManifest:
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


def _write_json_value(root: Path, relative: str, value: list[object]) -> None:
    destination = resolve_under(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(canonical_json(value) + b"\n")


def _validate_world_result(recipe: WorldRecipe, world: LatentWorldManifest, private: Path) -> None:
    if world.schema_version != "v7.2.latent-world.v1":
        raise PackageBuildError("family returned a legacy latent-world schema")
    _require_model_identity(world, "world_id", world.expected_world_id())
    if world.recipe_id != recipe.recipe_id or world.family_revision_id != recipe.family_revision_id:
        raise PackageBuildError("latent world does not bind the selected recipe and family")
    WorldStateView(private, world)


def _validate_graph_workflow(
    recipe: WorldRecipe,
    world: LatentWorldManifest,
    graph: ArtifactGraph,
    workflow: WorkflowGraph,
) -> None:
    if graph.schema_version != "v7.2.artifact-graph.v1":
        raise PackageBuildError("family returned a legacy artifact-graph schema")
    if workflow.schema_version != "v7.2.workflow-graph.v1":
        raise PackageBuildError("family returned a legacy workflow schema")
    _require_model_identity(
        graph,
        "graph_id",
        content_id("artifact-graph", graph.model_dump(mode="json", exclude={"graph_id"})),
    )
    _require_model_identity(
        workflow,
        "workflow_id",
        content_id(
            "workflow-graph",
            workflow.model_dump(mode="json", exclude={"workflow_id"}),
        ),
    )
    if graph.world_id != world.world_id:
        raise PackageBuildError("artifact graph does not bind the selected latent world")
    if workflow.artifact_graph_id != graph.graph_id:
        raise PackageBuildError("workflow does not bind the final artifact graph")
    if workflow.curriculum_layer != recipe.curriculum_layer:
        raise PackageBuildError("workflow curriculum layer differs from the world recipe")
    validate_artifact_graph(graph)
    validate_workflow_graph(workflow, graph)


def _validate_corruption_results(
    world: LatentWorldManifest,
    corruptions: tuple[CorruptionEvent, ...],
    public_render_root: Path,
) -> str:
    if not corruptions:
        raise PackageBuildError("family package requires a registered corruption event")
    if len({event.corruption_id for event in corruptions}) != len(corruptions):
        raise PackageBuildError("family returned duplicate corruption identities")
    for index, event in enumerate(corruptions):
        if event.schema_version != "v7.2.corruption-event.v1":
            raise PackageBuildError("family returned a legacy corruption schema")
        _require_model_identity(event, "corruption_id", event.expected_corruption_id())
        if event.world_id != world.world_id:
            raise PackageBuildError("corruption event does not bind the selected latent world")
        if index and event.baseline_tree_sha256 != corruptions[index - 1].corrupted_tree_sha256:
            raise PackageBuildError("corruption event tree identities do not form a sequence")
    if corruptions[-1].corrupted_tree_sha256 != tree_sha256(public_render_root):
        raise PackageBuildError("final public artifact tree differs from corruption evidence")
    return content_id(
        "v72-corruption-manifest",
        [event.model_dump(mode="json") for event in corruptions],
    )


class PackageBuilder:
    """Execute all Section 8 stages and atomically publish a task package."""

    def __init__(self, family: FamilyImplementation) -> None:
        self._family = family

    def build(
        self,
        *,
        recipe: WorldRecipe,
        source_records: tuple[SourceRecord, ...],
        source_payloads: tuple[SourcePayloadInput, ...],
        destination: Path,
        offline_prompt_authoring: OfflinePromptAuthoringInput | None = None,
    ) -> PackageBuildResult:
        if recipe.schema_version != "v7.2.world-recipe.v1":
            raise PackageBuildError("package builder accepts only v7.2 world recipes")
        _require_model_identity(recipe, "recipe_id", recipe.expected_recipe_id())
        if not source_records:
            raise PackageBuildError("package builder requires source governance records")
        ordered_sources = tuple(sorted(source_records, key=lambda record: record.source_id))
        if len({record.source_id for record in ordered_sources}) != len(ordered_sources):
            raise PackageBuildError("source record identities must be unique")
        for record in ordered_sources:
            if record.schema_version != "v7.2.source-record.v1":
                raise PackageBuildError("package builder rejects legacy source records")
            _require_model_identity(record, "source_id", record.expected_source_id())
        if tuple(record.source_id for record in ordered_sources) != recipe.source_record_ids:
            raise PackageBuildError("source records do not exactly match the world recipe")
        payloads_by_source = {item.authority.source_id: item for item in source_payloads}
        if len(payloads_by_source) != len(source_payloads) or set(payloads_by_source) != set(
            recipe.source_record_ids
        ):
            raise PackageBuildError("source payloads do not cover the exact world recipe")
        records_by_source = {item.source_id: item for item in ordered_sources}
        for source_id, payload in payloads_by_source.items():
            source_root = payload.root.resolve(strict=True)
            record = records_by_source[source_id]
            if (
                payload.root.is_symlink()
                or not source_root.is_dir()
                or not iter_regular_files(source_root)
                or payload.authority.source_tree_sha256 != record.source_tree_sha256
                or payload.authority.acquisition_record_id != record.acquisition_record_id
                or tree_sha256(source_root) != record.source_tree_sha256
            ):
                raise PackageBuildError("source payload bytes differ from governance authority")
            if record.source_kind == "generated":
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
                    raise PackageBuildError(
                        "generated source payload does not match its deterministic authority"
                    )
        try:
            family_revision = build_family_revision_authority(
                self._family,
                recipe.family_revision_id,
            )
        except (ImportError, OSError, TypeError, ValueError) as error:
            raise PackageBuildError(f"family revision authority is incomplete: {error}") from error
        for item in source_payloads:
            record = records_by_source[item.authority.source_id]
            expected_realization = (
                family_revision.materializer_authority.authority_id
                if item.authority.realization_kind == "deterministic_generator"
                else record.acquisition_record_id
            )
            if item.authority.realization_authority_id != expected_realization:
                raise PackageBuildError(
                    "source payload realization does not bind its exact producer authority"
                )

        result_values: dict[str, object] = {}
        with tempfile.TemporaryDirectory(prefix="v72-package-scratch-") as scratch_raw:
            scratch_public = Path(scratch_raw) / "public"
            scratch_public.mkdir()
            (scratch_public / "input").mkdir()
            (scratch_public / "output").mkdir()

            with atomic_directory(destination, prefix="v72-task-package-") as stage:
                private_root = stage / "private"
                private_root.mkdir()
                governed_source_root = private_root / "governed_sources"
                governed_source_root.mkdir()
                for source_id, payload in sorted(payloads_by_source.items()):
                    source = payload.root.resolve(strict=True)
                    target = governed_source_root / source_id.rsplit(":", 1)[1]
                    target.mkdir()
                    for path in iter_regular_files(source):
                        destination_path = target / path.relative_to(source)
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, destination_path)
                make_tree_read_only(governed_source_root)
                payload_authorities = tuple(
                    payloads_by_source[source_id].authority
                    for source_id in recipe.source_record_ids
                )
                governed_sources = GovernedSourceView(
                    private_root=private_root,
                    source_records=ordered_sources,
                    payload_authorities=payload_authorities,
                )
                world = self._family.build_world(recipe, private_root, governed_sources)
                if not isinstance(world, LatentWorldManifest):
                    raise PackageBuildError("build_world did not return LatentWorldManifest")
                _validate_world_result(recipe, world, private_root)

                workflow = self._family.build_workflow(world)
                if not isinstance(workflow, WorkflowGraph):
                    raise PackageBuildError("build_workflow did not return WorkflowGraph")
                baseline_graph = self._family.render_artifacts(
                    world,
                    workflow,
                    scratch_public,
                    private_root,
                )
                if not isinstance(baseline_graph, ArtifactGraph):
                    raise PackageBuildError("render_artifacts did not return ArtifactGraph")
                rendered_roots = tuple(sorted(path.name for path in scratch_public.iterdir()))
                if rendered_roots not in (("input", "output"), ("input", "output", "software")):
                    raise PackageBuildError(
                        "family rendered paths outside public input/output/software"
                    )
                if iter_regular_files(scratch_public / "output"):
                    raise PackageBuildError("family render must not prepopulate policy output")

                graph, corruptions = self._family.apply_corruptions(
                    world,
                    baseline_graph,
                    scratch_public,
                )
                if not isinstance(graph, ArtifactGraph) or type(corruptions) is not tuple:
                    raise PackageBuildError("apply_corruptions returned incomplete authority")
                _validate_world_result(recipe, world, private_root)
                _validate_graph_workflow(recipe, world, graph, workflow)
                corruption_manifest_id = _validate_corruption_results(
                    world,
                    corruptions,
                    scratch_public,
                )

                specification = self._family.build_public_specification(graph, workflow)
                if not isinstance(specification, PublicTaskSpecification):
                    raise PackageBuildError(
                        "build_public_specification did not return PublicTaskSpecification"
                    )
                if specification.schema_version != "v7.2.public-task.v1":
                    raise PackageBuildError("family returned a legacy public-task schema")
                _require_model_identity(
                    specification,
                    "specification_id",
                    specification.expected_specification_id(),
                )
                reward_contract = None
                build_reward_contract = getattr(self._family, "build_reward_contract", None)
                if callable(build_reward_contract):
                    reward_contract = build_reward_contract()
                    if not isinstance(reward_contract, RewardContract):
                        raise PackageBuildError(
                            "build_reward_contract did not return RewardContract"
                        )
                    _require_model_identity(
                        reward_contract,
                        "contract_id",
                        reward_contract.expected_contract_id(),
                    )
                progress_contract = None
                build_progress_contract = getattr(self._family, "build_progress_contract", None)
                if callable(build_progress_contract):
                    progress_contract = build_progress_contract(graph, workflow)
                    if not isinstance(progress_contract, ProgressContract):
                        raise PackageBuildError(
                            "build_progress_contract did not return ProgressContract"
                        )
                    _require_model_identity(
                        progress_contract,
                        "progress_contract_id",
                        progress_contract.expected_progress_contract_id(),
                    )
                    if reward_contract is None:
                        raise PackageBuildError(
                            "progress authority requires final reward authority"
                        )
                    if (
                        progress_contract.workflow_id != workflow.workflow_id
                        or progress_contract.family_revision_id != recipe.family_revision_id
                        or progress_contract.final_reward_contract_id != reward_contract.contract_id
                    ):
                        raise PackageBuildError("progress authority has stale family bindings")
                public_workspace, frozen_input = render_public_authority(
                    specification,
                    scratch_public / "input",
                    stage / "public",
                    software_source=(
                        scratch_public / "software"
                        if (scratch_public / "software").is_dir()
                        else None
                    ),
                )

                hidden_suite = self._family.build_hidden_suite(recipe, world, private_root)
                if not isinstance(hidden_suite, HiddenSuiteManifest):
                    raise PackageBuildError("build_hidden_suite did not return HiddenSuiteManifest")
                _require_model_identity(
                    hidden_suite,
                    "hidden_suite_id",
                    hidden_suite.expected_hidden_suite_id(),
                )
                if hidden_suite.world_id != world.world_id:
                    raise PackageBuildError("hidden suite does not bind the selected world")
                hidden_seeds = tuple(case.hidden_seed for case in hidden_suite.case_records)
                if len(set(hidden_seeds)) != len(hidden_seeds) or set(hidden_seeds).intersection(
                    {
                        recipe.public_seed,
                        recipe.private_world_seed,
                        recipe.perturbation_seed,
                        recipe.behavior_seed,
                    }
                ):
                    raise PackageBuildError("hidden case seeds are aliased or duplicated")
                validate_hidden_suite(private_root / "hidden_suite", hidden_suite)
                frozen_input.require_unchanged(stage / "public" / "input")
                if offline_prompt_authoring is not None:
                    specification = freeze_offline_authored_prompt(
                        specification,
                        offline_prompt_authoring,
                        private_roots=(
                            private_root / "world",
                            private_root / "hidden_suite",
                            private_root / "governed_sources",
                        ),
                        public_roots=tuple(
                            path
                            for path in (
                                scratch_public / "input",
                                scratch_public / "software",
                            )
                            if path.is_dir()
                        ),
                    )
                    make_tree_writable(stage / "public")
                    shutil.rmtree(stage / "public")
                    public_workspace, frozen_input = render_public_authority(
                        specification,
                        scratch_public / "input",
                        stage / "public",
                        software_source=(
                            scratch_public / "software"
                            if (scratch_public / "software").is_dir()
                            else None
                        ),
                    )
                prompt_generation_authority = build_prompt_generation_authority(
                    specification,
                    transcript=(
                        offline_prompt_authoring.transcript
                        if offline_prompt_authoring is not None
                        else None
                    ),
                )
                if (
                    file_sha256(stage / "public" / "prompt.md")
                    != prompt_generation_authority.prompt_sha256
                ):
                    raise PackageBuildError(
                        "rendered prompt differs from prompt generation authority"
                    )
                _validate_world_result(recipe, world, private_root)

                authority_root = private_root / "authority"
                authority_root.mkdir()
                write_authority_model(authority_root, "recipe.json", recipe)
                write_authority_model(authority_root, "world_manifest.json", world)
                write_authority_model(authority_root, "artifact_graph.json", graph)
                write_authority_model(authority_root, "workflow.json", workflow)
                write_authority_model(authority_root, "hidden_suite_manifest.json", hidden_suite)
                write_authority_model(
                    authority_root,
                    "prompt_generation.json",
                    prompt_generation_authority,
                )
                write_authority_model(authority_root, "public_specification.json", specification)
                if reward_contract is not None:
                    write_authority_model(authority_root, "reward_contract.json", reward_contract)
                if progress_contract is not None:
                    write_authority_model(
                        authority_root,
                        "progress_contract.json",
                        progress_contract,
                    )
                _write_json_value(
                    authority_root,
                    "corruption_events.json",
                    [event.model_dump(mode="json") for event in corruptions],
                )
                _write_json_value(
                    authority_root,
                    "source_records.json",
                    [record.model_dump(mode="json") for record in ordered_sources],
                )
                _write_json_value(
                    authority_root,
                    "source_payload_authorities.json",
                    [item.model_dump(mode="json") for item in payload_authorities],
                )
                if (
                    tuple(sorted(path.name for path in private_root.iterdir()))
                    != PRIVATE_ROOT_PATHS
                ):
                    raise PackageBuildError("private package tree has undeclared top-level paths")
                expected_authority_paths = tuple(
                    sorted(
                        (*PRIVATE_AUTHORITY_PATHS,)
                        + (("reward_contract.json",) if reward_contract is not None else ())
                        + (("progress_contract.json",) if progress_contract is not None else ())
                    )
                )
                if (
                    tuple(sorted(path.name for path in authority_root.iterdir()))
                    != expected_authority_paths
                ):
                    raise PackageBuildError("private authority tree is incomplete")

                public_manifest = _file_manifest(stage / "public", "public")
                private_manifest = _file_manifest(private_root, "private")
                policy_digest_set_id = content_id(
                    "v72-policy-digest-set",
                    [binding.model_dump(mode="json") for binding in recipe.policy_digest_set],
                )
                task_manifest = TaskPackageManifest.identified(
                    family_revision=family_revision,
                    recipe_id=recipe.recipe_id,
                    world_id=world.world_id,
                    artifact_graph_id=graph.graph_id,
                    workflow_id=workflow.workflow_id,
                    corruption_manifest_id=corruption_manifest_id,
                    hidden_suite_id=hidden_suite.hidden_suite_id,
                    public_specification_id=specification.specification_id,
                    prompt_sha256=file_sha256(stage / "public" / "prompt.md"),
                    public_manifest_id=public_manifest.manifest_id,
                    private_manifest_id=private_manifest.manifest_id,
                    public_tree_sha256=public_manifest.tree_sha256,
                    private_tree_sha256=private_manifest.tree_sha256,
                    public_input_tree_sha256=public_workspace.input_tree_sha256,
                    scale_id=recipe.scale_id,
                    curriculum_layer=recipe.curriculum_layer,
                    behavior_seed=recipe.behavior_seed,
                    source_record_ids=recipe.source_record_ids,
                    policy_digest_set_id=policy_digest_set_id,
                    reward_contract_id=(
                        reward_contract.contract_id if reward_contract is not None else None
                    ),
                    progress_contract_id=(
                        progress_contract.progress_contract_id
                        if progress_contract is not None
                        else None
                    ),
                )
                write_authority_model(stage, "package_manifest.json", task_manifest)
                write_authority_model(stage, "public_manifest.json", public_manifest)
                write_authority_model(stage, "private_manifest.json", private_manifest)
                if tuple(sorted(path.name for path in stage.iterdir())) != PACKAGE_ROOT_PATHS:
                    raise PackageBuildError("task package root layout is not exact")

                result_values = {
                    "task_manifest": task_manifest,
                    "public_file_manifest": public_manifest,
                    "private_file_manifest": private_manifest,
                    "public_workspace_manifest": public_workspace,
                    "frozen_public_input": frozen_input,
                    "world_manifest": world,
                    "artifact_graph": graph,
                    "workflow": workflow,
                    "corruptions": corruptions,
                    "hidden_suite": hidden_suite,
                    "public_specification": specification,
                    "prompt_generation_authority": prompt_generation_authority,
                    "source_records": ordered_sources,
                    "source_payload_authorities": payload_authorities,
                    "reward_contract": reward_contract,
                    "progress_contract": progress_contract,
                }

        return PackageBuildResult(package_root=destination, **result_values)  # type: ignore[arg-type]


__all__ = [
    "PACKAGE_ROOT_PATHS",
    "PRIVATE_AUTHORITY_PATHS",
    "PRIVATE_OPTIONAL_AUTHORITY_PATHS",
    "PRIVATE_ROOT_PATHS",
    "PackageBuildError",
    "PackageBuildResult",
    "PackageBuilder",
    "SourcePayloadInput",
    "create_generated_source_payload_input",
    "stage_generated_source_view",
]
