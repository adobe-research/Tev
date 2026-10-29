"""Causal DAG and critical-stage qualification for v7.2 workflows."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from itertools import pairwise
from typing import Literal, Self

from pydantic import model_validator

from training_data_collection.common.identity import canonical_json, content_id
from training_data_collection.v72.contracts.artifact import ArtifactGraph
from training_data_collection.v72.contracts.identity import ContentIdentifier, StrictFrozenModel
from training_data_collection.v72.contracts.workflow import WorkflowGraph, WorkflowStage

type AblationKind = Literal["stage_omission", "stale_intermediate"]
type StageAblationHook = Callable[
    [WorkflowStage, WorkflowGraph, ArtifactGraph, AblationKind], object
]


class StageAblationObservation(StrictFrozenModel):
    schema_version: Literal["v7.2.stage-ablation-observation.v1"]
    observation_id: ContentIdentifier
    stage_id: str
    ablation_kind: AblationKind
    changed_downstream_stage_ids: tuple[str, ...]
    changed_artifact_ids: tuple[ContentIdentifier, ...]
    lost_reward_component_ids: tuple[str, ...]
    verifier_evidence_ids: tuple[ContentIdentifier, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"observation_id"})
        if self.observation_id != content_id("stage-ablation", authority):
            raise ValueError("observation_id does not match stage ablation authority")
        return self

    @classmethod
    def create(
        cls,
        *,
        stage_id: str,
        ablation_kind: AblationKind,
        changed_downstream_stage_ids: tuple[str, ...],
        changed_artifact_ids: tuple[str, ...],
        lost_reward_component_ids: tuple[str, ...],
        verifier_evidence_ids: tuple[str, ...],
    ) -> Self:
        authority = {
            "schema_version": "v7.2.stage-ablation-observation.v1",
            "stage_id": stage_id,
            "ablation_kind": ablation_kind,
            "changed_downstream_stage_ids": list(changed_downstream_stage_ids),
            "changed_artifact_ids": list(changed_artifact_ids),
            "lost_reward_component_ids": list(lost_reward_component_ids),
            "verifier_evidence_ids": list(verifier_evidence_ids),
        }
        return cls.model_validate_json(
            canonical_json({"observation_id": content_id("stage-ablation", authority), **authority})
        )


class WorkflowQualificationEvidence(StrictFrozenModel):
    schema_version: Literal["v7.2.workflow-qualification.v1"]
    evidence_id: ContentIdentifier
    workflow_id: ContentIdentifier
    artifact_graph_id: ContentIdentifier
    semantic_critical_path: int
    stage_ablation_observations: tuple[StageAblationObservation, ...]
    qualified: Literal[True]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority = self.model_dump(mode="json", exclude={"evidence_id"})
        if self.evidence_id != content_id("workflow-qualification", authority):
            raise ValueError("evidence_id does not match workflow qualification authority")
        return self


class StageAblationRegistry:
    def __init__(self) -> None:
        self._hooks: dict[tuple[str, AblationKind], StageAblationHook] = {}

    def register(
        self,
        implementation_id: str,
        ablation_kind: AblationKind,
        hook: StageAblationHook,
    ) -> None:
        key = (implementation_id, ablation_kind)
        if not implementation_id:
            raise ValueError("ablation implementation ID must not be empty")
        if key in self._hooks:
            raise ValueError(f"duplicate stage ablation hook: {key}")
        self._hooks[key] = hook

    def resolve(self, implementation_id: str, ablation_kind: AblationKind) -> StageAblationHook:
        try:
            return self._hooks[(implementation_id, ablation_kind)]
        except KeyError as error:
            raise ValueError(f"unregistered {ablation_kind} hook: {implementation_id}") from error


def _topological_order(stages: dict[str, WorkflowStage]) -> tuple[str, ...]:
    indegree = {stage_id: 0 for stage_id in stages}
    dependents: dict[str, set[str]] = defaultdict(set)
    for stage in stages.values():
        for dependency_id in stage.depends_on_stage_ids:
            if dependency_id not in stages:
                raise ValueError(f"workflow stage has unknown dependency: {dependency_id}")
            indegree[stage.stage_id] += 1
            dependents[dependency_id].add(stage.stage_id)
    queue = deque(sorted(stage_id for stage_id, count in indegree.items() if count == 0))
    order: list[str] = []
    while queue:
        stage_id = queue.popleft()
        order.append(stage_id)
        for dependent_id in sorted(dependents.get(stage_id, set())):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                queue.append(dependent_id)
    if len(order) != len(stages):
        raise ValueError("workflow graph contains a dependency cycle")
    return tuple(order)


def _ancestor_sets(stages: dict[str, WorkflowStage], order: tuple[str, ...]) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {}
    for stage_id in order:
        direct = set(stages[stage_id].depends_on_stage_ids)
        ancestors[stage_id] = direct | {
            ancestor_id for dependency_id in direct for ancestor_id in ancestors[dependency_id]
        }
    return ancestors


def _descendant_sets(stages: dict[str, WorkflowStage]) -> dict[str, set[str]]:
    dependents: dict[str, set[str]] = defaultdict(set)
    for stage in stages.values():
        for dependency_id in stage.depends_on_stage_ids:
            dependents[dependency_id].add(stage.stage_id)
    descendants: dict[str, set[str]] = {}
    for stage_id in stages:
        queue = deque(dependents.get(stage_id, set()))
        reached: set[str] = set()
        while queue:
            dependent_id = queue.popleft()
            if dependent_id in reached:
                continue
            reached.add(dependent_id)
            queue.extend(dependents.get(dependent_id, set()))
        descendants[stage_id] = reached
    return descendants


def validate_workflow_graph(workflow: WorkflowGraph, artifact_graph: ArtifactGraph) -> None:
    if workflow.artifact_graph_id != artifact_graph.graph_id:
        raise ValueError("workflow does not bind the supplied artifact graph")
    if len(set(workflow.initial_state_ids)) != len(workflow.initial_state_ids):
        raise ValueError("workflow initial state IDs must be unique")
    if len(set(workflow.terminal_output_artifact_ids)) != len(
        workflow.terminal_output_artifact_ids
    ):
        raise ValueError("workflow terminal output artifact IDs must be unique")
    stages = {stage.stage_id: stage for stage in workflow.stages}
    if not stages or len(stages) != len(workflow.stages):
        raise ValueError("workflow requires unique stages")
    evidence = {item.evidence_id: item for item in workflow.verifier_evidence}
    if len(evidence) != len(workflow.verifier_evidence):
        raise ValueError("workflow contains duplicate verifier evidence IDs")
    artifacts = {node.artifact_id: node for node in artifact_graph.nodes}
    relations = {relation.relation_id: relation for relation in artifact_graph.relations}
    order = _topological_order(stages)
    ancestors = _ancestor_sets(stages, order)
    critical_ids = workflow.critical_stage_ids
    if not critical_ids or len(set(critical_ids)) != len(critical_ids):
        raise ValueError("workflow requires unique critical stage IDs")
    if not set(critical_ids) <= set(stages):
        raise ValueError("workflow critical path references unknown stages")
    if workflow.curriculum_layer in {"family_integration", "professional_frontier"} and not (
        3 <= len(critical_ids) <= 6
    ):
        raise ValueError("integration and frontier critical paths require 3 to 6 stages")
    for prior_id, stage_id in pairwise(critical_ids):
        if prior_id not in stages[stage_id].depends_on_stage_ids:
            raise ValueError("critical stages must form a connected dependency path")

    produced_states: dict[str, str] = {}
    produced_artifacts: dict[str, str] = {}
    for stage in workflow.stages:
        unknown_artifacts = set(stage.consumed_artifact_ids + stage.produced_artifact_ids) - set(
            artifacts
        )
        if unknown_artifacts:
            raise ValueError(
                f"workflow stage references unknown artifacts: {sorted(unknown_artifacts)}"
            )
        unknown_relations = set(stage.consumed_relation_ids) - set(relations)
        if unknown_relations:
            raise ValueError(
                f"workflow stage references unknown relations: {sorted(unknown_relations)}"
            )
        unknown_evidence = set(stage.verifier_evidence_ids) - set(evidence)
        if unknown_evidence:
            raise ValueError(
                f"workflow stage references unknown evidence: {sorted(unknown_evidence)}"
            )
        for state_id in stage.produced_state_ids:
            if state_id in produced_states or state_id in workflow.initial_state_ids:
                raise ValueError(f"workflow state has multiple authorities: {state_id}")
            produced_states[state_id] = stage.stage_id
        for artifact_id in stage.produced_artifact_ids:
            if artifact_id in produced_artifacts:
                raise ValueError(f"workflow artifact has multiple producing stages: {artifact_id}")
            produced_artifacts[artifact_id] = stage.stage_id

    all_state_ids = set(workflow.initial_state_ids) | set(produced_states)
    for stage in workflow.stages:
        for state_id in stage.input_state_ids:
            if state_id in workflow.initial_state_ids:
                continue
            producer_id = produced_states.get(state_id)
            if producer_id is None:
                raise ValueError(f"workflow stage consumes unknown state: {state_id}")
            if producer_id not in ancestors[stage.stage_id]:
                raise ValueError("workflow stage consumes state outside its dependency ancestry")
        for transition in stage.expected_failure_transitions:
            if (
                transition.from_state_id not in all_state_ids
                or transition.to_state_id not in all_state_ids
            ):
                raise ValueError("failure transition references unknown workflow state")
        for relation_id in stage.consumed_relation_ids:
            if stage.stage_id not in relations[relation_id].workflow_stage_ids:
                raise ValueError("stage and artifact relation workflow bindings disagree")
        for artifact_id in stage.consumed_artifact_ids:
            producer_id = produced_artifacts.get(artifact_id)
            if producer_id is not None and producer_id not in ancestors[stage.stage_id]:
                raise ValueError(
                    "workflow stage consumes an artifact outside its dependency ancestry"
                )

    for relation in artifact_graph.relations:
        unknown_stages = set(relation.workflow_stage_ids) - set(stages)
        if unknown_stages:
            raise ValueError(
                f"artifact relation references unknown workflow stages: {unknown_stages}"
            )
        for stage_id in relation.workflow_stage_ids:
            if relation.relation_id not in stages[stage_id].consumed_relation_ids:
                raise ValueError("artifact relation and stage consumption bindings disagree")

    for stage_id in critical_ids:
        stage = stages[stage_id]
        if not stage.reward_component_ids:
            raise ValueError("critical stages require reward component bindings")
        independent_evidence = [evidence[item] for item in stage.verifier_evidence_ids]
        if not independent_evidence or any(
            item.observation_kind == "self_report" or not item.independently_recomputed
            for item in independent_evidence
        ):
            raise ValueError("critical stages require independently recomputed verifier evidence")
        for item in independent_evidence:
            observed = set(item.observed_artifact_ids).intersection(stage.produced_artifact_ids)
            observed.update(set(item.observed_state_ids).intersection(stage.produced_state_ids))
            if not observed:
                raise ValueError("critical-stage evidence does not observe stage output")

    terminal_ids = set(workflow.terminal_output_artifact_ids)
    if terminal_ids != set(artifact_graph.terminal_output_artifact_ids):
        raise ValueError("workflow terminal outputs disagree with the artifact graph")
    if not terminal_ids <= set(produced_artifacts):
        raise ValueError("workflow terminal output is unreachable from any stage")
    final_critical_stage = stages[critical_ids[-1]]
    if not terminal_ids <= set(final_critical_stage.produced_artifact_ids):
        raise ValueError("final critical stage must produce every terminal output")

    seen_parallel_groups: set[frozenset[str]] = set()
    for group in workflow.allowed_parallel_groups:
        members = frozenset(group.stage_ids)
        if members in seen_parallel_groups or not members <= set(stages):
            raise ValueError("parallel group is duplicate or references an unknown stage")
        seen_parallel_groups.add(members)
        if any(
            left in ancestors[right] or right in ancestors[left]
            for left in members
            for right in members
            if left != right
        ):
            raise ValueError("dependency-related stages cannot be declared parallel")


def qualify_workflow_graph(
    workflow: WorkflowGraph,
    artifact_graph: ArtifactGraph,
    ablation_registry: StageAblationRegistry,
) -> WorkflowQualificationEvidence:
    validate_workflow_graph(workflow, artifact_graph)
    stages = {stage.stage_id: stage for stage in workflow.stages}
    descendants = _descendant_sets(stages)
    evidence_ids = {item.evidence_id for item in workflow.verifier_evidence}
    observations: list[StageAblationObservation] = []
    for stage_id in workflow.critical_stage_ids:
        stage = stages[stage_id]
        descendant_rewards = {
            reward_id
            for descendant_id in descendants[stage_id] | {stage_id}
            for reward_id in stages[descendant_id].reward_component_ids
        }
        for ablation_kind in ("stage_omission", "stale_intermediate"):
            hook = ablation_registry.resolve(stage.ablation_implementation_id, ablation_kind)
            observation = hook(stage, workflow, artifact_graph, ablation_kind)
            if not isinstance(observation, StageAblationObservation):
                raise TypeError("stage ablation hook must return StageAblationObservation")
            if observation.stage_id != stage_id or observation.ablation_kind != ablation_kind:
                raise ValueError("stage ablation observation does not match the requested ablation")
            if not set(observation.changed_downstream_stage_ids) <= descendants[stage_id]:
                raise ValueError("stage ablation names a non-downstream affected stage")
            if not set(observation.changed_artifact_ids) <= set(
                workflow.terminal_output_artifact_ids
            ):
                raise ValueError("stage ablation names an artifact outside terminal authority")
            if not (observation.changed_downstream_stage_ids or observation.changed_artifact_ids):
                raise ValueError("stage ablation did not change downstream observable state")
            if not set(observation.lost_reward_component_ids).intersection(descendant_rewards):
                raise ValueError("stage ablation did not affect a bound reward component")
            if (
                not observation.verifier_evidence_ids
                or not set(observation.verifier_evidence_ids) <= set(stage.verifier_evidence_ids)
                or not set(observation.verifier_evidence_ids) <= evidence_ids
            ):
                raise ValueError("stage ablation lacks independent verifier evidence")
            observations.append(observation)

    authority = {
        "schema_version": "v7.2.workflow-qualification.v1",
        "workflow_id": workflow.workflow_id,
        "artifact_graph_id": artifact_graph.graph_id,
        "semantic_critical_path": len(workflow.critical_stage_ids),
        "stage_ablation_observations": [
            observation.model_dump(mode="json") for observation in observations
        ],
        "qualified": True,
    }
    return WorkflowQualificationEvidence.model_validate_json(
        canonical_json(
            {"evidence_id": content_id("workflow-qualification", authority), **authority}
        )
    )
