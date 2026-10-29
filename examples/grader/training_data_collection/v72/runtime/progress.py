"""Immutable snapshot capture and incremental progress-DAG evaluation."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from training_data_collection.common.filesystem import (
    atomic_directory,
    file_sha256,
    iter_regular_files,
    resolve_under,
    tree_manifest,
    tree_sha256,
)
from training_data_collection.common.identity import content_digest, content_id
from training_data_collection.v72.contracts.progress import (
    ProgressCheckpoint,
    ProgressContract,
    ProgressEvent,
    ProgressNode,
    ProgressObservation,
    WorkspaceSnapshot,
)


@dataclass(frozen=True)
class CheckerMeasurement:
    """Finite semantic value and content-bound evidence returned by a node checker."""

    raw_value: float
    evidence_record_ids: tuple[str, ...]


class ProgressEvaluationError(RuntimeError):
    """Typed checker failure that must not become policy reward zero."""

    def __init__(self, kind: str, detail: str) -> None:
        if kind not in {"infrastructure_error", "task_defect", "verifier_error"}:
            raise ValueError("unknown progress evaluation error kind")
        if not detail:
            raise ValueError("progress evaluation errors require detail")
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


type ProgressChecker = Callable[[Path, Path, Path], CheckerMeasurement]


def _selector_digest(output_root: Path, selector: str) -> str:
    relative = selector.removeprefix("output/")
    if selector == "output":
        return _snapshot_tree_sha256(output_root)
    target = resolve_under(output_root, relative)
    if not target.exists():
        return content_digest({"selector": selector, "status": "missing"})
    if target.is_symlink():
        raise ProgressEvaluationError("verifier_error", "candidate output contains a symlink")
    if target.is_file():
        return content_digest(
            {"sha256": file_sha256(target), "mode": target.stat().st_mode & 0o777}
        )
    if target.is_dir():
        return _snapshot_tree_sha256(target)
    raise ProgressEvaluationError("verifier_error", "candidate output has unsupported state")


def _snapshot_tree_sha256(root: Path) -> str:
    return content_digest(
        [
            {**record, "mode": path.stat().st_mode & 0o777}
            for path, record in zip(iter_regular_files(root), tree_manifest(root), strict=True)
        ]
    )


def _topological_nodes(contract: ProgressContract) -> tuple[ProgressNode, ...]:
    by_id = {node.node_id: node for node in contract.nodes}
    ordered: list[ProgressNode] = []
    visited: set[str] = set()

    def visit(node: ProgressNode) -> None:
        if node.node_id in visited:
            return
        for prerequisite in node.prerequisite_node_ids:
            visit(by_id[prerequisite])
        visited.add(node.node_id)
        ordered.append(node)

    for node in contract.nodes:
        visit(node)
    return tuple(ordered)


class SnapshotStore:
    """Verifier-owned content-addressed snapshots of policy output state."""

    def __init__(self, root: Path) -> None:
        if root.exists():
            if root.is_symlink() or not root.is_dir():
                raise ValueError("snapshot store must be a real directory")
        else:
            root.mkdir(parents=True, exist_ok=False)
        self.root = root.resolve(strict=True)

    def capture(
        self,
        *,
        episode_id: str,
        output_root: Path,
        checkpoint_index: int,
        action_index: int | None,
        parent_snapshot: WorkspaceSnapshot | None,
        selectors: tuple[str, ...],
        execution_state_digest: str | None = None,
    ) -> WorkspaceSnapshot:
        """Seal one immutable output tree and bind every relevant selector digest."""
        source = output_root.resolve(strict=True)
        if source.is_symlink() or not source.is_dir():
            raise ValueError("policy output root must be a real directory")
        tree_digest = _snapshot_tree_sha256(source)
        relevant = {
            selector: _selector_digest(source, selector) for selector in sorted(set(selectors))
        }
        storage_reference = f"objects/{tree_digest}"
        object_root = self.root / storage_reference
        if not object_root.exists():
            with atomic_directory(object_root, prefix="v72-progress-snapshot-") as stage:
                for path in iter_regular_files(source):
                    destination = stage / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    destination.chmod(path.stat().st_mode & 0o777)
                if _snapshot_tree_sha256(stage) != tree_digest:
                    raise ValueError("captured snapshot differs from policy output")
        elif _snapshot_tree_sha256(object_root) != tree_digest:
            raise ValueError("snapshot object content is stale")
        return WorkspaceSnapshot.identified(
            episode_id=episode_id,
            checkpoint_index=checkpoint_index,
            action_index=action_index,
            parent_snapshot_id=(
                parent_snapshot.snapshot_id if parent_snapshot is not None else None
            ),
            workspace_tree_sha256=tree_digest,
            relevant_state_digests=relevant,
            execution_state_digest=execution_state_digest,
            storage_reference=storage_reference,
        )

    def resolve(self, snapshot: WorkspaceSnapshot) -> Path:
        """Resolve and verify one immutable snapshot object."""
        root = resolve_under(self.root, snapshot.storage_reference)
        if _snapshot_tree_sha256(root) != snapshot.workspace_tree_sha256:
            raise ValueError("stored workspace snapshot differs from its authority")
        return root


class ProgressEvaluator:
    """Evaluate complete vectors while caching only exact checker input authority."""

    def __init__(
        self,
        *,
        contract: ProgressContract,
        checker_registry: Mapping[str, ProgressChecker],
        snapshot_store: SnapshotStore,
        public_root: Path,
        private_root: Path,
        target_authority_id: str,
        runtime_identity: str,
    ) -> None:
        expected_checkers = {node.checker_implementation_id for node in contract.nodes}
        if set(checker_registry) != expected_checkers:
            raise ValueError("progress checker registry must exactly cover the contract")
        self.contract = contract
        self.checker_registry = dict(checker_registry)
        self.snapshot_store = snapshot_store
        self.public_root = public_root.resolve(strict=True)
        self.private_root = private_root.resolve(strict=True)
        self.target_authority_id = target_authority_id
        self.runtime_identity = runtime_identity
        self.public_tree_sha256 = tree_sha256(self.public_root)
        self._cache: dict[str, CheckerMeasurement] = {}

    @property
    def selectors(self) -> tuple[str, ...]:
        return tuple(
            sorted({selector for node in self.contract.nodes for selector in node.input_selectors})
        )

    def capture_initial(self, *, episode_id: str, output_root: Path) -> WorkspaceSnapshot:
        return self.snapshot_store.capture(
            episode_id=episode_id,
            output_root=output_root,
            checkpoint_index=0,
            action_index=None,
            parent_snapshot=None,
            selectors=self.selectors,
        )

    def capture_after_action(
        self,
        *,
        episode_id: str,
        output_root: Path,
        action_index: int,
        previous: WorkspaceSnapshot,
    ) -> WorkspaceSnapshot:
        candidate_digest = _snapshot_tree_sha256(output_root)
        if candidate_digest == previous.workspace_tree_sha256:
            return previous
        return self.snapshot_store.capture(
            episode_id=episode_id,
            output_root=output_root,
            checkpoint_index=previous.checkpoint_index + 1,
            action_index=action_index,
            parent_snapshot=previous,
            selectors=self.selectors,
        )

    def evaluate(
        self,
        snapshot: WorkspaceSnapshot,
        *,
        use_cache: bool = True,
    ) -> ProgressCheckpoint:
        """Build a complete raw and effective vector for one immutable state."""
        output_root = self.snapshot_store.resolve(snapshot)
        observations_by_id: dict[str, ProgressObservation] = {}
        effective: dict[str, float | None] = {}
        raw: dict[str, float | None] = {}
        weights = {node.node_id: node.weight for node in self.contract.nodes}
        for node in _topological_nodes(self.contract):
            prerequisite_values = {
                node_id: effective[node_id]
                for node_id in node.prerequisite_node_ids
                if effective[node_id] is not None
            }
            input_digest = content_digest(
                {
                    "checker_implementation_id": node.checker_implementation_id,
                    "progress_contract_id": self.contract.progress_contract_id,
                    "selectors": {
                        selector: snapshot.relevant_state_digests[selector]
                        for selector in node.input_selectors
                    },
                    "public_tree_sha256": self.public_tree_sha256,
                    "target_authority_id": self.target_authority_id,
                    "runtime_identity": self.runtime_identity,
                }
            )
            prerequisite_error = len(prerequisite_values) != len(node.prerequisite_node_ids)
            hard_blocked = node.gate_policy == "hard" and (
                prerequisite_error
                or any(
                    value < node.prerequisite_threshold for value in prerequisite_values.values()
                )
            )
            if hard_blocked:
                observation = ProgressObservation.identified(
                    progress_contract_id=self.contract.progress_contract_id,
                    snapshot_id=snapshot.snapshot_id,
                    node_id=node.node_id,
                    status="blocked",
                    raw_value=None,
                    effective_value=0.0,
                    prerequisite_values=prerequisite_values,
                    evidence_record_ids=(),
                    input_digest=input_digest,
                    checker_duration_seconds=0.0,
                )
            else:
                started = time.monotonic()
                try:
                    measurement = self._cache.get(input_digest) if use_cache else None
                    status = "cached" if measurement is not None else "evaluated"
                    if measurement is None:
                        measurement = self.checker_registry[node.checker_implementation_id](
                            self.public_root,
                            output_root,
                            self.private_root,
                        )
                        if (
                            not 0.0 <= measurement.raw_value <= 1.0
                            or measurement.raw_value != measurement.raw_value
                        ):
                            raise ProgressEvaluationError(
                                "verifier_error", "progress checker returned a non-finite fraction"
                            )
                        self._cache[input_digest] = measurement
                    effective_value = measurement.raw_value
                    if node.gate_policy == "smooth_cap":
                        effective_value = min(
                            effective_value,
                            min(prerequisite_values.values(), default=0.0),
                        )
                    observation = ProgressObservation.identified(
                        progress_contract_id=self.contract.progress_contract_id,
                        snapshot_id=snapshot.snapshot_id,
                        node_id=node.node_id,
                        status=status,
                        raw_value=measurement.raw_value,
                        effective_value=effective_value,
                        prerequisite_values=prerequisite_values,
                        evidence_record_ids=tuple(sorted(set(measurement.evidence_record_ids))),
                        input_digest=input_digest,
                        checker_duration_seconds=time.monotonic() - started,
                    )
                except ProgressEvaluationError as error:
                    observation = ProgressObservation.identified(
                        progress_contract_id=self.contract.progress_contract_id,
                        snapshot_id=snapshot.snapshot_id,
                        node_id=node.node_id,
                        status="error",
                        raw_value=None,
                        effective_value=None,
                        prerequisite_values=prerequisite_values,
                        evidence_record_ids=(),
                        input_digest=input_digest,
                        checker_duration_seconds=time.monotonic() - started,
                        error_kind=error.kind,
                        error_detail=error.detail,
                    )
            observations_by_id[node.node_id] = observation
            raw[node.node_id] = observation.raw_value
            effective[node.node_id] = observation.effective_value
        observations = tuple(observations_by_id[node.node_id] for node in self.contract.nodes)
        has_error = any(item.status == "error" for item in observations)
        potential = None
        if not has_error:
            potential = sum(
                weights[node_id] * (value or 0.0) for node_id, value in effective.items()
            )
        return ProgressCheckpoint.identified(
            progress_contract_id=self.contract.progress_contract_id,
            snapshot_id=snapshot.snapshot_id,
            observations=observations,
            raw_vector={node.node_id: raw[node.node_id] for node in self.contract.nodes},
            effective_vector={
                node.node_id: effective[node.node_id] for node in self.contract.nodes
            },
            potential=potential,
            error_observation_ids=tuple(
                item.observation_id for item in observations if item.status == "error"
            ),
        )

    def transition(
        self,
        *,
        episode_id: str,
        action_index: int,
        before_snapshot: WorkspaceSnapshot,
        before_checkpoint: ProgressCheckpoint,
        output_root: Path,
    ) -> tuple[WorkspaceSnapshot, ProgressCheckpoint, ProgressEvent]:
        """Capture and attribute the semantic delta caused by one policy action."""
        after_snapshot = self.capture_after_action(
            episode_id=episode_id,
            output_root=output_root,
            action_index=action_index,
            previous=before_snapshot,
        )
        after_checkpoint = (
            before_checkpoint
            if after_snapshot.snapshot_id == before_snapshot.snapshot_id
            else self.evaluate(after_snapshot)
        )
        improved: list[str] = []
        regressed: list[str] = []
        for node_id in before_checkpoint.effective_vector:
            before_value = before_checkpoint.effective_vector[node_id]
            after_value = after_checkpoint.effective_vector[node_id]
            if before_value is None or after_value is None:
                continue
            if after_value > before_value:
                improved.append(node_id)
            elif after_value < before_value:
                regressed.append(node_id)
        shaping_reward = None
        if before_checkpoint.potential is not None and after_checkpoint.potential is not None:
            shaping_reward = self.contract.shaping_scale * (
                after_checkpoint.potential - before_checkpoint.potential
            )
        evidence = tuple(
            sorted(
                {
                    evidence_id
                    for checkpoint in (before_checkpoint, after_checkpoint)
                    for observation in checkpoint.observations
                    for evidence_id in observation.evidence_record_ids
                }
            )
        )
        event = ProgressEvent.identified(
            episode_id=episode_id,
            progress_contract_id=self.contract.progress_contract_id,
            action_index=action_index,
            from_snapshot_id=before_snapshot.snapshot_id,
            to_snapshot_id=after_snapshot.snapshot_id,
            from_checkpoint_id=before_checkpoint.checkpoint_id,
            to_checkpoint_id=after_checkpoint.checkpoint_id,
            raw_vector_before=before_checkpoint.raw_vector,
            raw_vector_after=after_checkpoint.raw_vector,
            effective_vector_before=before_checkpoint.effective_vector,
            effective_vector_after=after_checkpoint.effective_vector,
            potential_before=before_checkpoint.potential,
            potential_after=after_checkpoint.potential,
            shaping_reward=shaping_reward,
            improved_node_ids=tuple(sorted(improved)),
            regressed_node_ids=tuple(sorted(regressed)),
            evidence_record_ids=evidence,
        )
        return after_snapshot, after_checkpoint, event


def snapshot_content_evidence(root: Path) -> str:
    """Bind paths, bytes, and modes for diagnostics outside the snapshot identity."""
    return content_id(
        "v72-snapshot-content-evidence",
        [
            {**record, "mode": path.stat().st_mode & 0o777}
            for path, record in zip(iter_regular_files(root), tree_manifest(root), strict=True)
        ],
    )


__all__ = [
    "CheckerMeasurement",
    "ProgressChecker",
    "ProgressEvaluationError",
    "ProgressEvaluator",
    "SnapshotStore",
    "snapshot_content_evidence",
]
