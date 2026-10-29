"""Structural and realized-artifact qualification for v7.2 artifact graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from training_data_collection.common.filesystem import (
    file_sha256,
    iter_regular_files,
    resolve_under,
    safe_relative_path,
    tree_sha256,
)
from training_data_collection.common.identity import canonical_json, content_digest, content_id
from training_data_collection.v72.contracts.artifact import ArtifactGraph, ArtifactRelation
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)

_DERIVATION_RELATION_TYPES = {
    "derived_from",
    "generated_from",
    "summarizes",
    "indexes",
    "shards",
    "reassembles",
    "joins_on",
    "aligns_with",
    "hashes",
    "constrained_by",
}


class RealizedArtifactDigest(StrictFrozenModel):
    artifact_id: ContentIdentifier
    digest: Sha256Digest


class RelationValidationObservation(StrictFrozenModel):
    """Exact realized endpoint authority returned by one semantic validator."""

    passed: Literal[True]
    observed_artifact_digests: tuple[RealizedArtifactDigest, ...]

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        artifact_ids = tuple(item.artifact_id for item in self.observed_artifact_digests)
        if not artifact_ids or artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ValueError("relation endpoint observations must be nonempty, sorted, and unique")
        return self


class RelationValidatorExecution(StrictFrozenModel):
    schema_version: Literal["v7.2.relation-validator-execution.v1"]
    execution_id: ContentIdentifier
    relation_id: ContentIdentifier
    validator_implementation_id: str
    accessed_artifact_ids: tuple[ContentIdentifier, ...]
    realized_artifact_digests: tuple[RealizedArtifactDigest, ...]
    passed: Literal[True]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"execution_id"})
        if self.execution_id != content_id("relation-execution", authority):
            raise ValueError("execution_id does not match relation execution authority")
        return self


class ArtifactGraphQualificationEvidence(StrictFrozenModel):
    schema_version: Literal["v7.2.artifact-graph-qualification.v1"]
    evidence_id: ContentIdentifier
    graph_id: ContentIdentifier
    realized_artifact_digests: tuple[RealizedArtifactDigest, ...]
    relation_executions: tuple[RelationValidatorExecution, ...]
    qualified: Literal[True]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"evidence_id"})
        if self.evidence_id != content_id("artifact-qualification", authority):
            raise ValueError("evidence_id does not match artifact qualification authority")
        return self


class ArtifactRelationValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, RelationValidator] = {}

    def register(self, implementation_id: str, validator: RelationValidator) -> None:
        if not implementation_id:
            raise ValueError("validator implementation ID must not be empty")
        if implementation_id in self._validators:
            raise ValueError(f"duplicate relation validator: {implementation_id}")
        self._validators[implementation_id] = validator

    def resolve(self, implementation_id: str) -> RelationValidator:
        try:
            return self._validators[implementation_id]
        except KeyError as error:
            raise ValueError(f"unregistered relation validator: {implementation_id}") from error


class RealizedArtifactView(Mapping[str, tuple[Path, ...]]):
    """Read-only realized paths plus content digests observed by relation validators."""

    def __init__(
        self,
        artifacts: Mapping[str, tuple[Path, ...]],
        digests: Mapping[str, RealizedArtifactDigest],
    ) -> None:
        self._artifacts = artifacts
        self._digests = digests
        self.accessed: set[str] = set()

    def __getitem__(self, artifact_id: str) -> tuple[Path, ...]:
        self.accessed.add(artifact_id)
        return self._artifacts[artifact_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._artifacts)

    def __len__(self) -> int:
        return len(self._artifacts)

    def observe(self, artifact_id: str) -> RealizedArtifactDigest:
        """Record access and return the recomputed digest for one realized artifact."""
        self.accessed.add(artifact_id)
        return self._digests[artifact_id]


type RelationValidator = Callable[
    [ArtifactRelation, RealizedArtifactView], RelationValidationObservation
]


def observe_relation_endpoints(
    relation: ArtifactRelation,
    realized: RealizedArtifactView,
    *,
    passed: bool,
) -> RelationValidationObservation:
    """Bind a successful semantic result to every exact realized relation endpoint."""
    if passed is not True:
        raise ValueError(f"artifact relation validator failed: {relation.relation_id}")
    endpoint_ids = tuple(sorted(set(relation.source_artifact_ids + relation.target_artifact_ids)))
    return RelationValidationObservation(
        passed=True,
        observed_artifact_digests=tuple(realized.observe(item) for item in endpoint_ids),
    )


def _has_symlink_component(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _strongly_connected_derivation_relations(
    graph: ArtifactGraph,
) -> tuple[frozenset[str], ...]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    relations_by_endpoints: dict[tuple[str, str], set[str]] = defaultdict(set)
    for relation in graph.relations:
        if relation.relation_type not in _DERIVATION_RELATION_TYPES:
            continue
        for source_id in relation.source_artifact_ids:
            for target_id in relation.target_artifact_ids:
                adjacency[source_id].add(target_id)
                relations_by_endpoints[(source_id, target_id)].add(relation.relation_id)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[set[str]] = []

    def visit(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for target_id in adjacency.get(node_id, set()):
            if target_id not in indices:
                visit(target_id)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target_id])
            elif target_id in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[target_id])
        if lowlinks[node_id] != indices[node_id]:
            return
        component: set[str] = set()
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.add(member)
            if member == node_id:
                break
        components.append(component)

    for artifact_id in {node.artifact_id for node in graph.nodes}:
        if artifact_id not in indices:
            visit(artifact_id)

    cyclic_relation_sets: list[frozenset[str]] = []
    for component in components:
        self_loop = any(node_id in adjacency.get(node_id, set()) for node_id in component)
        if len(component) == 1 and not self_loop:
            continue
        relation_ids = {
            relation_id
            for (source_id, target_id), ids in relations_by_endpoints.items()
            if source_id in component and target_id in component
            for relation_id in ids
        }
        cyclic_relation_sets.append(frozenset(relation_ids))
    return tuple(cyclic_relation_sets)


def validate_artifact_graph(graph: ArtifactGraph) -> None:
    nodes = {node.artifact_id: node for node in graph.nodes}
    relations = {relation.relation_id: relation for relation in graph.relations}
    if len(nodes) != len(graph.nodes):
        raise ValueError("artifact graph contains duplicate node IDs")
    if len(relations) != len(graph.relations):
        raise ValueError("artifact graph contains duplicate relation IDs")
    path_authorities = {(node.visibility, node.path) for node in graph.nodes}
    if len(path_authorities) != len(graph.nodes):
        raise ValueError("artifact graph contains duplicate visibility/path authorities")

    for relation in graph.relations:
        endpoints = set(relation.source_artifact_ids + relation.target_artifact_ids)
        unknown = endpoints - set(nodes)
        if unknown:
            raise ValueError(f"artifact relation has unknown endpoints: {sorted(unknown)}")
        source_ranks = {nodes[item].authority_rank for item in relation.source_artifact_ids}
        target_ranks = {nodes[item].authority_rank for item in relation.target_artifact_ids}
        if relation.relation_type == "contradicts" and not (
            min(source_ranks) > max(target_ranks) or min(target_ranks) > max(source_ranks)
        ):
            raise ValueError("contradictory artifacts require a resolvable authority rank")
        if relation.relation_type == "supersedes" and min(source_ranks) <= max(target_ranks):
            raise ValueError("superseding artifacts must have strictly higher authority")
        if relation.critical and not (
            relation.public_rule_ids
            and relation.invariant_ids
            and relation.workflow_stage_ids
            and relation.reward_component_ids
        ):
            raise ValueError(
                "critical relations require public, invariant, workflow, and reward bindings"
            )

    approval_sets = {frozenset(cycle.relation_ids) for cycle in graph.approved_derivation_cycles}
    if len(approval_sets) != len(graph.approved_derivation_cycles):
        raise ValueError("artifact graph contains duplicate derivation cycle approvals")
    for approval in graph.approved_derivation_cycles:
        if not set(approval.relation_ids) <= set(relations):
            raise ValueError("derivation cycle approval references an unknown relation")
        if any(
            relations[relation_id].relation_type not in _DERIVATION_RELATION_TYPES
            for relation_id in approval.relation_ids
        ):
            raise ValueError("derivation cycle approval includes a non-derivation relation")
    actual_cycle_sets = set(_strongly_connected_derivation_relations(graph))
    for cycle_relation_ids in actual_cycle_sets:
        if cycle_relation_ids not in approval_sets:
            raise ValueError("artifact graph contains an unapproved derivation cycle")
    if approval_sets - actual_cycle_sets:
        raise ValueError("derivation cycle approval does not bind an actual cycle")

    if len(set(graph.critical_public_artifact_ids)) != len(graph.critical_public_artifact_ids):
        raise ValueError("critical public artifact IDs must be unique")
    if len(set(graph.terminal_output_artifact_ids)) != len(graph.terminal_output_artifact_ids):
        raise ValueError("terminal output artifact IDs must be unique")
    critical_ids = set(graph.critical_public_artifact_ids)
    terminal_ids = set(graph.terminal_output_artifact_ids)
    if not critical_ids or not terminal_ids:
        raise ValueError("artifact graph requires critical public inputs and terminal outputs")
    if not critical_ids <= set(nodes) or not terminal_ids <= set(nodes):
        raise ValueError("critical or terminal artifact ID does not resolve")
    if any(nodes[item].visibility != "public" for item in critical_ids):
        raise ValueError("critical public artifact IDs must reference public artifacts")
    if any(nodes[item].semantic_role != "required_output" for item in terminal_ids):
        raise ValueError("terminal output IDs must reference required-output artifacts")
    if any(nodes[item].visibility != "public" for item in terminal_ids):
        raise ValueError("terminal output IDs must reference public artifacts")

    causal_adjacency: dict[str, set[str]] = defaultdict(set)
    for relation in graph.relations:
        if not (
            relation.public_rule_ids
            and relation.workflow_stage_ids
            and relation.reward_component_ids
        ):
            continue
        for source_id in relation.source_artifact_ids:
            causal_adjacency[source_id].update(relation.target_artifact_ids)
    for artifact_id in critical_ids:
        queue = deque([artifact_id])
        reached = {artifact_id}
        while queue:
            new_artifact_ids = causal_adjacency.get(queue.popleft(), set()) - reached
            reached.update(new_artifact_ids)
            queue.extend(new_artifact_ids)
        if not reached.intersection(terminal_ids):
            raise ValueError("critical public artifact has no causal route to a terminal reward")


def _realize_artifacts(
    graph: ArtifactGraph,
    public_root: Path,
    private_root: Path | None,
) -> tuple[dict[str, tuple[Path, ...]], tuple[RealizedArtifactDigest, ...]]:
    artifacts: dict[str, tuple[Path, ...]] = {}
    digests: list[RealizedArtifactDigest] = []
    for node in graph.nodes:
        root = public_root if node.visibility == "public" else private_root
        if root is None:
            raise ValueError("verifier-private artifacts require a private authority root")
        root = root.resolve(strict=True)
        if node.path_kind == "glob":
            paths = tuple(sorted(root.glob(node.path), key=lambda item: item.as_posix()))
            if not paths:
                raise ValueError(f"artifact glob did not resolve: {node.path}")
            if any(
                path.is_symlink()
                or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(root)
                or _has_symlink_component(root, path)
                for path in paths
            ):
                raise ValueError(f"artifact glob must resolve only regular files: {node.path}")
            digest = content_digest(
                [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": file_sha256(path),
                    }
                    for path in paths
                ]
            )
        else:
            unresolved = root.joinpath(*safe_relative_path(node.path).parts)
            if _has_symlink_component(root, unresolved):
                raise ValueError(f"artifact path traverses a symlink: {node.path}")
            path = resolve_under(root, node.path)
            if node.path_kind == "file":
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"artifact file did not resolve: {node.path}")
                paths = (path,)
                digest = file_sha256(path)
            else:
                iter_regular_files(path)
                paths = (path,)
                digest = tree_sha256(path)
        artifacts[node.artifact_id] = paths
        digests.append(RealizedArtifactDigest(artifact_id=node.artifact_id, digest=digest))
    return artifacts, tuple(digests)


def qualify_artifact_graph(
    graph: ArtifactGraph,
    public_root: Path,
    private_root: Path | None,
    registry: ArtifactRelationValidatorRegistry,
) -> ArtifactGraphQualificationEvidence:
    validate_artifact_graph(graph)
    artifacts, realized_digests = _realize_artifacts(graph, public_root, private_root)
    executions: list[RelationValidatorExecution] = []
    digests_by_id = {item.artifact_id: item for item in realized_digests}
    for relation in graph.relations:
        validator = registry.resolve(relation.validator_implementation_id)
        tracked = RealizedArtifactView(artifacts, digests_by_id)
        observation = validator(relation, tracked)
        if not isinstance(observation, RelationValidationObservation):
            raise TypeError("artifact relation validator must return RelationValidationObservation")
        endpoints = set(relation.source_artifact_ids + relation.target_artifact_ids)
        observed_digests = {
            item.artifact_id: item for item in observation.observed_artifact_digests
        }
        if set(observed_digests) != endpoints or any(
            observed_digests[item] != digests_by_id[item] for item in endpoints
        ):
            raise ValueError("relation validator endpoint digest authority is incomplete")
        if endpoints != tracked.accessed:
            raise ValueError("relation validator must inspect exactly its realized endpoints")
        authority = {
            "schema_version": "v7.2.relation-validator-execution.v1",
            "relation_id": relation.relation_id,
            "validator_implementation_id": relation.validator_implementation_id,
            "accessed_artifact_ids": sorted(tracked.accessed),
            "realized_artifact_digests": [
                digests_by_id[item].model_dump(mode="json") for item in sorted(endpoints)
            ],
            "passed": True,
        }
        executions.append(
            RelationValidatorExecution.model_validate_json(
                canonical_json(
                    {"execution_id": content_id("relation-execution", authority), **authority}
                )
            )
        )
    authority = {
        "schema_version": "v7.2.artifact-graph-qualification.v1",
        "graph_id": graph.graph_id,
        "realized_artifact_digests": [item.model_dump(mode="json") for item in realized_digests],
        "relation_executions": [item.model_dump(mode="json") for item in executions],
        "qualified": True,
    }
    return ArtifactGraphQualificationEvidence.model_validate_json(
        canonical_json(
            {"evidence_id": content_id("artifact-qualification", authority), **authority}
        )
    )


__all__ = [
    "ArtifactGraphQualificationEvidence",
    "ArtifactRelationValidatorRegistry",
    "RealizedArtifactDigest",
    "RealizedArtifactView",
    "RelationValidationObservation",
    "RelationValidatorExecution",
    "observe_relation_endpoints",
    "qualify_artifact_graph",
    "validate_artifact_graph",
]
