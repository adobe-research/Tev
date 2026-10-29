"""Action-level semantic progress authority for v7.2 tasks."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, ValidationInfo, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import content_id

from .identity import ContentIdentifier, Sha256Digest, StrictFrozenModel

NodeIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=160,
        pattern=r"^[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*$",
        strict=True,
    ),
]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, strict=True)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
NonNegativeInteger = Annotated[int, Field(ge=0)]


class ProgressNode(StrictFrozenModel):
    """One independently measurable semantic property in a progress DAG."""

    schema_version: Literal["v7.2.progress-node.v1"] = "v7.2.progress-node.v1"
    node_record_id: ContentIdentifier
    node_id: NodeIdentifier
    description: NonEmptyString
    observed_artifact_ids: tuple[NonEmptyString, ...] = ()
    observed_state_ids: tuple[NonEmptyString, ...] = ()
    prerequisite_node_ids: tuple[NodeIdentifier, ...] = ()
    gate_policy: Literal["independent", "hard", "smooth_cap"]
    prerequisite_threshold: Fraction = 1.0
    checker_implementation_id: ContentIdentifier
    continuous_score_definition: NonEmptyString
    weight: Fraction
    causal_mutation_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    allowed_collateral_node_ids: tuple[NodeIdentifier, ...] = ()
    input_selectors: tuple[NonEmptyString, ...] = Field(min_length=1)
    online_evaluation_cost: Literal["cheap", "moderate", "expensive"]
    public_rule_ids: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_node(self, info: ValidationInfo) -> Self:
        for label, values in (
            ("observed artifacts", self.observed_artifact_ids),
            ("observed states", self.observed_state_ids),
            ("prerequisites", self.prerequisite_node_ids),
            ("causal mutations", self.causal_mutation_ids),
            ("collateral nodes", self.allowed_collateral_node_ids),
            ("input selectors", self.input_selectors),
            ("public rules", self.public_rule_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"progress node {label} must be sorted and unique")
        if not self.observed_artifact_ids and not self.observed_state_ids:
            raise ValueError("progress nodes must observe an artifact or state")
        if self.node_id in self.prerequisite_node_ids:
            raise ValueError("progress nodes cannot depend on themselves")
        if self.node_id in self.allowed_collateral_node_ids:
            raise ValueError("progress nodes cannot declare themselves as collateral")
        if self.gate_policy == "independent" and self.prerequisite_node_ids:
            raise ValueError("independent progress nodes cannot have prerequisites")
        if self.gate_policy != "independent" and not self.prerequisite_node_ids:
            raise ValueError("gated progress nodes require prerequisites")
        for selector in self.input_selectors:
            if selector != "output" and not selector.startswith("output/"):
                raise ValueError("progress selectors must remain under the policy output root")
            safe_relative_path(selector)
        if not (info.context or {}).get("skip_identity") and (
            self.node_record_id != self.expected_node_record_id()
        ):
            raise ValueError("progress node identity is stale")
        return self

    def expected_node_record_id(self) -> str:
        return content_id(
            "v72-progress-node",
            self.model_dump(mode="json", exclude={"node_record_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"node_record_id": f"v72-progress-node:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            node_record_id=provisional.expected_node_record_id(),
            **provisional.model_dump(mode="python", exclude={"node_record_id"}),
        )


class ProgressContract(StrictFrozenModel):
    """Content-bound progress graph and shaping policy for one family revision."""

    schema_version: Literal["v7.2.progress-contract.v1"] = "v7.2.progress-contract.v1"
    progress_contract_id: ContentIdentifier
    family_revision_id: NonEmptyString
    workflow_id: ContentIdentifier
    final_reward_contract_id: ContentIdentifier
    nodes: tuple[ProgressNode, ...] = Field(min_length=1)
    potential_definition: Literal["weighted_effective_sum"] = "weighted_effective_sum"
    shaping_scale: Fraction
    snapshot_policy: Literal["after_relevant_mutation"] = "after_relevant_mutation"
    terminal_evaluation_policy: Literal["full_uncached"] = "full_uncached"

    @model_validator(mode="after")
    def validate_contract(self, info: ValidationInfo) -> Self:
        node_ids = tuple(node.node_id for node in self.nodes)
        if node_ids != tuple(sorted(set(node_ids))):
            raise ValueError("progress nodes must be sorted by unique node ID")
        known = set(node_ids)
        for node in self.nodes:
            if not set(node.prerequisite_node_ids).issubset(known):
                raise ValueError("progress node has an unknown prerequisite")
            if not set(node.allowed_collateral_node_ids).issubset(known):
                raise ValueError("progress node has unknown collateral effects")
        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {node.node_id: node for node in self.nodes}

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("progress prerequisites must form a DAG")
            if node_id in visited:
                return
            visiting.add(node_id)
            for prerequisite in by_id[node_id].prerequisite_node_ids:
                visit(prerequisite)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in node_ids:
            visit(node_id)
        total_weight = sum(node.weight for node in self.nodes)
        if not math.isfinite(total_weight) or total_weight > 1.0 + 1e-12:
            raise ValueError("progress potential must be bounded by one")
        if total_weight <= 0.0:
            raise ValueError("progress potential must contain positive shaping weight")
        if not any(node.weight == 0.0 for node in self.nodes):
            raise ValueError("progress contract must retain a zero-weight terminal diagnostic")
        if not (info.context or {}).get("skip_identity") and (
            self.progress_contract_id != self.expected_progress_contract_id()
        ):
            raise ValueError("progress contract identity is stale")
        return self

    def expected_progress_contract_id(self) -> str:
        return content_id(
            "v72-progress-contract",
            self.model_dump(mode="json", exclude={"progress_contract_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        if "nodes" in values:
            values["nodes"] = tuple(sorted(values["nodes"], key=lambda node: node.node_id))
        provisional = cls.model_validate(
            {"progress_contract_id": f"v72-progress-contract:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            progress_contract_id=provisional.expected_progress_contract_id(),
            **provisional.model_dump(mode="python", exclude={"progress_contract_id"}),
        )


class WorkspaceSnapshot(StrictFrozenModel):
    """Immutable content-addressed policy-output state at an action boundary."""

    schema_version: Literal["v7.2.workspace-snapshot.v1"] = "v7.2.workspace-snapshot.v1"
    snapshot_id: ContentIdentifier
    episode_id: ContentIdentifier
    checkpoint_index: NonNegativeInteger
    action_index: NonNegativeInteger | None
    parent_snapshot_id: ContentIdentifier | None
    workspace_tree_sha256: Sha256Digest
    relevant_state_digests: dict[NonEmptyString, Sha256Digest]
    execution_state_digest: Sha256Digest | None = None
    storage_reference: NonEmptyString

    @model_validator(mode="after")
    def validate_snapshot(self, info: ValidationInfo) -> Self:
        if tuple(self.relevant_state_digests) != tuple(sorted(self.relevant_state_digests)):
            raise ValueError("snapshot state digests must use sorted keys")
        safe_relative_path(self.storage_reference)
        if self.checkpoint_index == 0:
            if self.action_index is not None or self.parent_snapshot_id is not None:
                raise ValueError("initial snapshot cannot name an action or parent")
        elif self.action_index is None or self.parent_snapshot_id is None:
            raise ValueError("post-action snapshots require an action and parent")
        if not (info.context or {}).get("skip_identity") and (
            self.snapshot_id != self.expected_snapshot_id()
        ):
            raise ValueError("workspace snapshot identity is stale")
        return self

    def expected_snapshot_id(self) -> str:
        return content_id(
            "v72-workspace-snapshot",
            self.model_dump(mode="json", exclude={"snapshot_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"snapshot_id": f"v72-workspace-snapshot:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            snapshot_id=provisional.expected_snapshot_id(),
            **provisional.model_dump(mode="python", exclude={"snapshot_id"}),
        )


class ProgressObservation(StrictFrozenModel):
    """One node's raw and prerequisite-gated value at a checkpoint."""

    schema_version: Literal["v7.2.progress-observation.v1"] = "v7.2.progress-observation.v1"
    observation_id: ContentIdentifier
    progress_contract_id: ContentIdentifier
    snapshot_id: ContentIdentifier
    node_id: NodeIdentifier
    status: Literal["evaluated", "cached", "blocked", "not_applicable", "error"]
    raw_value: Fraction | None
    effective_value: Fraction | None
    prerequisite_values: dict[NodeIdentifier, Fraction]
    evidence_record_ids: tuple[ContentIdentifier, ...]
    input_digest: Sha256Digest
    checker_duration_seconds: NonNegativeFloat
    error_kind: Literal["infrastructure_error", "task_defect", "verifier_error"] | None = None
    error_detail: str | None = None

    @model_validator(mode="after")
    def validate_observation(self, info: ValidationInfo) -> Self:
        if tuple(self.prerequisite_values) != tuple(sorted(self.prerequisite_values)):
            raise ValueError("observation prerequisite values must use sorted keys")
        if self.evidence_record_ids != tuple(sorted(set(self.evidence_record_ids))):
            raise ValueError("observation evidence IDs must be sorted and unique")
        if self.status == "error":
            if self.raw_value is not None or self.effective_value is not None:
                raise ValueError("checker errors cannot be encoded as policy scores")
            if self.error_kind is None or not self.error_detail:
                raise ValueError("checker errors require typed failure detail")
        elif self.error_kind is not None or self.error_detail is not None:
            raise ValueError("healthy observations cannot carry checker failure detail")
        elif self.status in {"evaluated", "cached"} and (
            self.raw_value is None or self.effective_value is None
        ):
            raise ValueError("evaluated observations require raw and effective values")
        elif self.status in {"blocked", "not_applicable"} and self.effective_value != 0.0:
            raise ValueError("blocked observations must have zero effective value")
        if not (info.context or {}).get("skip_identity") and (
            self.observation_id != self.expected_observation_id()
        ):
            raise ValueError("progress observation identity is stale")
        return self

    def expected_observation_id(self) -> str:
        return content_id(
            "v72-progress-observation",
            self.model_dump(mode="json", exclude={"observation_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"observation_id": f"v72-progress-observation:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            observation_id=provisional.expected_observation_id(),
            **provisional.model_dump(mode="python", exclude={"observation_id"}),
        )


class ProgressCheckpoint(StrictFrozenModel):
    """Complete progress vector for one immutable snapshot."""

    schema_version: Literal["v7.2.progress-checkpoint.v1"] = "v7.2.progress-checkpoint.v1"
    checkpoint_id: ContentIdentifier
    progress_contract_id: ContentIdentifier
    snapshot_id: ContentIdentifier
    observations: tuple[ProgressObservation, ...]
    raw_vector: dict[NodeIdentifier, Fraction | None]
    effective_vector: dict[NodeIdentifier, Fraction | None]
    potential: Fraction | None
    error_observation_ids: tuple[ContentIdentifier, ...] = ()

    @model_validator(mode="after")
    def validate_checkpoint(self, info: ValidationInfo) -> Self:
        observation_ids = tuple(item.node_id for item in self.observations)
        if observation_ids != tuple(sorted(set(observation_ids))):
            raise ValueError("checkpoint observations must be sorted and complete")
        if (
            tuple(self.raw_vector) != observation_ids
            or tuple(self.effective_vector) != observation_ids
        ):
            raise ValueError("checkpoint vectors must exactly match observations")
        expected_errors = tuple(
            item.observation_id for item in self.observations if item.status == "error"
        )
        if self.error_observation_ids != expected_errors:
            raise ValueError("checkpoint error identities must be derived from observations")
        if (self.potential is None) != bool(expected_errors):
            raise ValueError("checkpoint potential must be absent exactly on checker error")
        if not (info.context or {}).get("skip_identity") and (
            self.checkpoint_id != self.expected_checkpoint_id()
        ):
            raise ValueError("progress checkpoint identity is stale")
        return self

    def expected_checkpoint_id(self) -> str:
        return content_id(
            "v72-progress-checkpoint",
            self.model_dump(mode="json", exclude={"checkpoint_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"checkpoint_id": f"v72-progress-checkpoint:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            checkpoint_id=provisional.expected_checkpoint_id(),
            **provisional.model_dump(mode="python", exclude={"checkpoint_id"}),
        )


class ProgressEvent(StrictFrozenModel):
    """Semantic potential change attributed to one state-changing policy action."""

    schema_version: Literal["v7.2.progress-event.v1"] = "v7.2.progress-event.v1"
    progress_event_id: ContentIdentifier
    episode_id: ContentIdentifier
    progress_contract_id: ContentIdentifier
    action_index: NonNegativeInteger
    from_snapshot_id: ContentIdentifier
    to_snapshot_id: ContentIdentifier
    from_checkpoint_id: ContentIdentifier
    to_checkpoint_id: ContentIdentifier
    raw_vector_before: dict[NodeIdentifier, Fraction | None]
    raw_vector_after: dict[NodeIdentifier, Fraction | None]
    effective_vector_before: dict[NodeIdentifier, Fraction | None]
    effective_vector_after: dict[NodeIdentifier, Fraction | None]
    potential_before: Fraction | None
    potential_after: Fraction | None
    shaping_reward: float | None = Field(allow_inf_nan=False)
    improved_node_ids: tuple[NodeIdentifier, ...]
    regressed_node_ids: tuple[NodeIdentifier, ...]
    evidence_record_ids: tuple[ContentIdentifier, ...]

    @model_validator(mode="after")
    def validate_event(self, info: ValidationInfo) -> Self:
        keys = tuple(self.raw_vector_before)
        vectors = (
            self.raw_vector_after,
            self.effective_vector_before,
            self.effective_vector_after,
        )
        if keys != tuple(sorted(keys)) or any(tuple(vector) != keys for vector in vectors):
            raise ValueError("progress event vectors must use the same sorted node keys")
        for label, values in (
            ("improved nodes", self.improved_node_ids),
            ("regressed nodes", self.regressed_node_ids),
            ("evidence", self.evidence_record_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"progress event {label} must be sorted and unique")
        if set(self.improved_node_ids) & set(self.regressed_node_ids):
            raise ValueError("one node cannot both improve and regress in one event")
        has_error = self.potential_before is None or self.potential_after is None
        if has_error != (self.shaping_reward is None):
            raise ValueError("progress shaping reward must be absent on evaluation error")
        if self.shaping_reward is not None and not math.isfinite(self.shaping_reward):
            raise ValueError("progress shaping reward must be finite")
        if not (info.context or {}).get("skip_identity") and (
            self.progress_event_id != self.expected_progress_event_id()
        ):
            raise ValueError("progress event identity is stale")
        return self

    def expected_progress_event_id(self) -> str:
        return content_id(
            "v72-progress-event",
            self.model_dump(mode="json", exclude={"progress_event_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"progress_event_id": f"v72-progress-event:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            progress_event_id=provisional.expected_progress_event_id(),
            **provisional.model_dump(mode="python", exclude={"progress_event_id"}),
        )


class ActionRewardAssignment(StrictFrozenModel):
    """Immediate reward and return assigned only to one assistant action span."""

    schema_version: Literal["v7.2.action-reward-assignment.v1"] = "v7.2.action-reward-assignment.v1"
    assignment_id: ContentIdentifier
    episode_id: ContentIdentifier
    action_index: NonNegativeInteger
    progress_event_id: ContentIdentifier | None
    assistant_token_start: NonNegativeInteger
    assistant_token_end: NonNegativeInteger
    immediate_shaping_reward: float = Field(allow_inf_nan=False)
    terminal_reward: Fraction
    discounted_return: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_assignment(self, info: ValidationInfo) -> Self:
        if self.assistant_token_end <= self.assistant_token_start:
            raise ValueError("assistant token spans must be nonempty")
        if not math.isfinite(self.immediate_shaping_reward) or not math.isfinite(
            self.discounted_return
        ):
            raise ValueError("action rewards and returns must be finite")
        if not (info.context or {}).get("skip_identity") and (
            self.assignment_id != self.expected_assignment_id()
        ):
            raise ValueError("action reward assignment identity is stale")
        return self

    def expected_assignment_id(self) -> str:
        return content_id(
            "v72-action-reward-assignment",
            self.model_dump(mode="json", exclude={"assignment_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"assignment_id": f"v72-action-reward-assignment:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            assignment_id=provisional.expected_assignment_id(),
            **provisional.model_dump(mode="python", exclude={"assignment_id"}),
        )


def build_action_reward_assignments(
    *,
    episode_id: str,
    action_token_spans: tuple[tuple[int, int], ...],
    progress_events: tuple[ProgressEvent | None, ...],
    terminal_reward: float,
    discount: float,
) -> tuple[ActionRewardAssignment, ...]:
    """Compute action-level discounted returns without broadcasting across tool tokens."""
    if len(action_token_spans) != len(progress_events) or not action_token_spans:
        raise ValueError("every action must have one token span and progress-event slot")
    if not 0.0 <= terminal_reward <= 1.0 or not 0.0 <= discount <= 1.0:
        raise ValueError("terminal reward and discount must be finite fractions")
    immediate = [
        0.0 if event is None or event.shaping_reward is None else event.shaping_reward
        for event in progress_events
    ]
    immediate[-1] += terminal_reward
    returns = [0.0] * len(immediate)
    running = 0.0
    for index in range(len(immediate) - 1, -1, -1):
        running = immediate[index] + discount * running
        returns[index] = running
    return tuple(
        ActionRewardAssignment.identified(
            episode_id=episode_id,
            action_index=index,
            progress_event_id=event.progress_event_id if event is not None else None,
            assistant_token_start=span[0],
            assistant_token_end=span[1],
            immediate_shaping_reward=(
                0.0 if event is None or event.shaping_reward is None else event.shaping_reward
            ),
            terminal_reward=terminal_reward if index == len(immediate) - 1 else 0.0,
            discounted_return=returns[index],
        )
        for index, (span, event) in enumerate(zip(action_token_spans, progress_events, strict=True))
    )


__all__ = [
    "ActionRewardAssignment",
    "ProgressCheckpoint",
    "ProgressContract",
    "ProgressEvent",
    "ProgressNode",
    "ProgressObservation",
    "WorkspaceSnapshot",
    "build_action_reward_assignments",
]
