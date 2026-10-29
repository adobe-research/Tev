"""Deterministic and transactional latent-world materialization."""

from __future__ import annotations

import ast
import inspect
import random
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

from training_data_collection.common.filesystem import (
    atomic_directory,
    resolve_under,
    safe_relative_path,
    tree_sha256,
)
from training_data_collection.v72.contracts.identity import SeedNamespace
from training_data_collection.v72.contracts.world import (
    LatentWorldManifest,
    WorldEntityState,
    WorldInvariant,
    WorldRecipe,
)

_FORBIDDEN_AUTHORITY_MODULES = frozenset(
    {
        "aiohttp",
        "datetime",
        "httpx",
        "os",
        "random",
        "requests",
        "secrets",
        "socket",
        "time",
        "urllib",
    }
)
_FORBIDDEN_AUTHORITY_CALLS = frozenset(
    {"__import__", "eval", "exec", "open", "time", "time_ns", "urlopen"}
)


class WorldGenerationError(ValueError):
    """Raised when latent-world construction violates closed-world authority."""


@dataclass(frozen=True)
class WorldBuildResult:
    """Private files and declarations returned by a family materializer."""

    files: Mapping[str, bytes]
    entity_index: tuple[WorldEntityState, ...]
    invariants: tuple[WorldInvariant, ...]
    renderer_declarations: tuple[str, ...]


@dataclass(frozen=True)
class WorldMaterialization:
    """Manifest and automatically recorded inputs for one realized world."""

    manifest: LatentWorldManifest
    requested_seed_namespaces: tuple[str, ...]
    requested_source_record_ids: tuple[str, ...]
    invariant_ids_validated: tuple[str, ...]


class WorldBuildContext:
    """The only random and governed-source authority exposed to a materializer."""

    def __init__(
        self,
        recipe: WorldRecipe,
        governed_sources: Mapping[str, bytes],
    ) -> None:
        if set(governed_sources) != set(recipe.source_record_ids):
            raise WorldGenerationError(
                "governed source bytes must exactly match recipe source identities"
            )
        if any(type(value) is not bytes for value in governed_sources.values()):
            raise WorldGenerationError("governed source values must be bytes")
        self._recipe = recipe
        self._governed_sources = dict(governed_sources)
        self._requested_namespaces: set[SeedNamespace] = set()
        self._requested_sources: set[str] = set()

    def seed(self, namespace: SeedNamespace) -> int:
        if type(namespace) is not SeedNamespace:
            raise WorldGenerationError("materializers must request a fixed seed namespace")
        if namespace is not SeedNamespace.PRIVATE_WORLD:
            raise WorldGenerationError(
                "latent-world materializers may use only the private-world seed namespace"
            )
        self._requested_namespaces.add(namespace)
        return self._recipe.private_world_seed

    def random(self, namespace: SeedNamespace) -> random.Random:
        return random.Random(self.seed(namespace))

    def read_source(self, source_record_id: str) -> bytes:
        try:
            value = self._governed_sources[source_record_id]
        except KeyError as error:
            raise WorldGenerationError(
                f"materializer requested undeclared source: {source_record_id}"
            ) from error
        self._requested_sources.add(source_record_id)
        return value

    @property
    def requested_seed_namespaces(self) -> tuple[str, ...]:
        return tuple(sorted(namespace.value for namespace in self._requested_namespaces))

    @property
    def requested_source_record_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._requested_sources))


class WorldMaterializer(Protocol):
    """Family-owned latent-world implementation boundary."""

    implementation_digest: str

    def build(
        self,
        recipe: WorldRecipe,
        context: WorldBuildContext,
    ) -> WorldBuildResult: ...


type WorldInvariantValidator = Callable[[Path, WorldInvariant], bool]


def audit_materializer_authority(materializer: WorldMaterializer) -> None:
    """Reject direct random, clock, filesystem, or network authority in a builder."""
    build = materializer.build
    try:
        source = textwrap.dedent(inspect.getsource(build))
    except (OSError, TypeError) as error:
        raise WorldGenerationError("materializer source must be inspectable") from error
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(module.split(".")[0] in _FORBIDDEN_AUTHORITY_MODULES for module in modules):
                raise WorldGenerationError(
                    "materializer imports forbidden random, clock, filesystem, or network authority"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_AUTHORITY_CALLS
        ):
            raise WorldGenerationError(f"materializer calls forbidden authority: {node.func.id}")

    closure = inspect.getclosurevars(build)
    for value in (*closure.globals.values(), *closure.nonlocals.values()):
        if isinstance(value, ModuleType) and value.__name__.split(".")[0] in (
            _FORBIDDEN_AUTHORITY_MODULES
        ):
            raise WorldGenerationError(
                f"materializer closes over forbidden authority module: {value.__name__}"
            )


def materialize_world(
    *,
    recipe: WorldRecipe,
    materializer: WorldMaterializer,
    destination: Path,
    invariant_validators: Mapping[str, WorldInvariantValidator],
    governed_sources: Mapping[str, bytes] | None = None,
) -> WorldMaterialization:
    """Build, validate, and atomically publish one private latent-world tree."""
    audit_materializer_authority(materializer)
    context = WorldBuildContext(recipe, governed_sources or {})
    result = materializer.build(recipe, context)
    if not isinstance(result, WorldBuildResult):
        raise WorldGenerationError("materializer must return WorldBuildResult")
    if context.requested_seed_namespaces != (SeedNamespace.PRIVATE_WORLD.value,):
        raise WorldGenerationError("materializer did not consume the private-world seed")
    if context.requested_source_record_ids != recipe.source_record_ids:
        raise WorldGenerationError("materializer did not consume every governed recipe source")
    if not result.files:
        raise WorldGenerationError("latent world state tree cannot be empty")

    files: dict[str, bytes] = {}
    for relative, value in result.files.items():
        normalized = safe_relative_path(relative).as_posix()
        if normalized in files:
            raise WorldGenerationError(f"duplicate world state path: {normalized}")
        if type(value) is not bytes:
            raise WorldGenerationError("world state values must be bytes")
        files[normalized] = value

    declared_paths = {entity.state_path for entity in result.entity_index}
    if declared_paths != set(files):
        raise WorldGenerationError(
            "world entity index must cover every realized state file exactly once"
        )
    invariant_validator_ids = {
        invariant.validator_implementation_id for invariant in result.invariants
    }
    missing_validators = invariant_validator_ids - set(invariant_validators)
    if missing_validators:
        raise WorldGenerationError(
            f"world invariants lack executable validators: {sorted(missing_validators)}"
        )

    with atomic_directory(destination, prefix="v72-world-") as stage:
        for relative, value in sorted(files.items()):
            target = resolve_under(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)

        state_digest = tree_sha256(stage)
        invariant_ids: list[str] = []
        for invariant in sorted(result.invariants, key=lambda item: item.invariant_id):
            validator = invariant_validators[invariant.validator_implementation_id]
            if validator(stage, invariant) is not True:
                raise WorldGenerationError(
                    f"realized world invariant failed: {invariant.invariant_id}"
                )
            invariant_ids.append(invariant.invariant_id)
        if tree_sha256(stage) != state_digest:
            raise WorldGenerationError("world invariant validator mutated private state")

        manifest = LatentWorldManifest.identified(
            recipe_id=recipe.recipe_id,
            family_revision_id=recipe.family_revision_id,
            entity_index=tuple(sorted(result.entity_index, key=lambda item: item.entity_id)),
            invariant_declarations=tuple(
                sorted(result.invariants, key=lambda item: item.invariant_id)
            ),
            state_tree_sha256=state_digest,
            renderer_declarations=tuple(sorted(result.renderer_declarations)),
            source_record_ids=recipe.source_record_ids,
            creation_implementation_digest=materializer.implementation_digest,
        )

    if tree_sha256(destination) != manifest.state_tree_sha256:
        raise WorldGenerationError("published world state differs from validated staging tree")
    return WorldMaterialization(
        manifest=manifest,
        requested_seed_namespaces=context.requested_seed_namespaces,
        requested_source_record_ids=context.requested_source_record_ids,
        invariant_ids_validated=tuple(invariant_ids),
    )


__all__ = [
    "WorldBuildContext",
    "WorldBuildResult",
    "WorldGenerationError",
    "WorldInvariantValidator",
    "WorldMaterialization",
    "WorldMaterializer",
    "audit_materializer_authority",
    "materialize_world",
]
