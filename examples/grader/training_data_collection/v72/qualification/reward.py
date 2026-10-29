"""Executed causal-mutation qualification for v7.2 reward components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import model_validator

from training_data_collection.common.identity import canonical_json, content_id
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)
from training_data_collection.v72.contracts.reward import (
    RewardComponent,
    RewardContract,
    RewardResult,
)


@dataclass(frozen=True)
class RewardMutationExecution:
    """Real outputs returned by one registered causal-mutation execution hook."""

    baseline_result: RewardResult
    mutated_result: RewardResult
    baseline_output_sha256: str
    mutated_output_sha256: str
    execution_evidence_ids: tuple[str, ...]


type RewardMutationHook = Callable[[RewardComponent], RewardMutationExecution]


class RewardMutationEvidence(StrictFrozenModel):
    """Content-bound proof that one mutation caused one component to lose credit."""

    schema_version: Literal["v7.2.reward-mutation-evidence.v1"] = "v7.2.reward-mutation-evidence.v1"
    evidence_id: ContentIdentifier
    mutation_id: str
    mutation_implementation_digest: Sha256Digest
    affected_component_id: str
    task_id: ContentIdentifier
    reward_contract_id: ContentIdentifier
    baseline_reward_record_id: ContentIdentifier
    mutated_reward_record_id: ContentIdentifier
    baseline_output_sha256: Sha256Digest
    mutated_output_sha256: Sha256Digest
    baseline_component_value: float
    mutated_component_value: float
    changed_component_ids: tuple[str, ...]
    allowed_collateral_component_ids: tuple[str, ...]
    execution_evidence_ids: tuple[ContentIdentifier, ...]
    independently_recomputed: Literal[True]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.baseline_reward_record_id == self.mutated_reward_record_id:
            raise ValueError("reward mutation must change the reward record")
        if self.baseline_output_sha256 == self.mutated_output_sha256:
            raise ValueError("reward mutation must change sealed output authority")
        if self.mutated_component_value >= self.baseline_component_value:
            raise ValueError("reward mutation must lower its affected component")
        if self.changed_component_ids != tuple(sorted(set(self.changed_component_ids))):
            raise ValueError("changed reward components must be sorted and unique")
        if self.allowed_collateral_component_ids != tuple(
            sorted(set(self.allowed_collateral_component_ids))
        ):
            raise ValueError("allowed collateral components must be sorted and unique")
        if self.affected_component_id not in self.changed_component_ids:
            raise ValueError("reward mutation must change its affected component")
        undeclared = set(self.changed_component_ids) - {
            self.affected_component_id,
            *self.allowed_collateral_component_ids,
        }
        if undeclared:
            raise ValueError(
                f"reward mutation changed undeclared collateral components: {sorted(undeclared)}"
            )
        if not self.execution_evidence_ids or len(self.execution_evidence_ids) != len(
            set(self.execution_evidence_ids)
        ):
            raise ValueError("reward mutation requires unique execution evidence")
        if self.evidence_id != self.expected_evidence_id():
            raise ValueError("reward mutation evidence identity is stale")
        return self

    def expected_evidence_id(self) -> str:
        return content_id(
            "v72-reward-mutation-evidence",
            self.model_dump(mode="json", exclude={"evidence_id"}),
        )


class RewardCausalityEvidence(StrictFrozenModel):
    """Complete mutation coverage for every component in one reward contract."""

    schema_version: Literal["v7.2.reward-causality-evidence.v1"] = (
        "v7.2.reward-causality-evidence.v1"
    )
    evidence_id: ContentIdentifier
    reward_contract_id: ContentIdentifier
    mutation_evidence: tuple[RewardMutationEvidence, ...]
    qualified: Literal[True]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        identities = tuple(item.evidence_id for item in self.mutation_evidence)
        if not identities or identities != tuple(sorted(identities)):
            raise ValueError("reward mutation evidence must be nonempty and sorted")
        if len(identities) != len(set(identities)):
            raise ValueError("reward mutation evidence identities must be unique")
        if any(
            item.reward_contract_id != self.reward_contract_id for item in self.mutation_evidence
        ):
            raise ValueError("reward mutation evidence mixes reward contracts")
        if self.evidence_id != self.expected_evidence_id():
            raise ValueError("reward causality evidence identity is stale")
        return self

    def expected_evidence_id(self) -> str:
        return content_id(
            "v72-reward-causality-evidence",
            self.model_dump(mode="json", exclude={"evidence_id"}),
        )


class RewardMutationRegistry:
    """Exact mutation implementations available to reward qualification."""

    def __init__(self) -> None:
        self._hooks: dict[str, tuple[str, RewardMutationHook]] = {}

    def register(
        self,
        mutation_id: str,
        implementation_digest: str,
        hook: RewardMutationHook,
    ) -> None:
        if not mutation_id.strip():
            raise ValueError("reward mutation identity cannot be blank")
        if mutation_id in self._hooks:
            raise ValueError(f"duplicate reward mutation implementation: {mutation_id}")
        if len(implementation_digest) != 64 or any(
            character not in "0123456789abcdef" for character in implementation_digest
        ):
            raise ValueError("reward mutation implementation digest must be lowercase SHA-256")
        self._hooks[mutation_id] = (implementation_digest, hook)

    def resolve(self, mutation_id: str) -> tuple[str, RewardMutationHook]:
        try:
            return self._hooks[mutation_id]
        except KeyError as error:
            raise ValueError(f"unregistered reward mutation: {mutation_id}") from error


def _validate_execution(
    contract: RewardContract,
    component: RewardComponent,
    mutation_id: str,
    implementation_digest: str,
    execution: RewardMutationExecution,
) -> RewardMutationEvidence:
    if not isinstance(execution, RewardMutationExecution):
        raise TypeError("reward mutation hook must return RewardMutationExecution")
    baseline = execution.baseline_result
    mutated = execution.mutated_result
    if baseline.status != "policy_result" or mutated.status != "policy_result":
        raise ValueError("reward mutations cannot encode system failures as policy outcomes")
    if baseline.task_id != mutated.task_id:
        raise ValueError("reward mutation results must bind the same task")
    if (
        baseline.reward_contract_id != contract.contract_id
        or mutated.reward_contract_id != contract.contract_id
    ):
        raise ValueError("reward mutation results do not bind the selected contract")
    expected_components = {item.component_id for item in contract.components}
    if (
        set(baseline.component_values) != expected_components
        or set(mutated.component_values) != expected_components
    ):
        raise ValueError("reward mutation result component sets are incomplete")
    baseline_value = baseline.component_values[component.component_id]
    mutated_value = mutated.component_values[component.component_id]
    changed_component_ids = tuple(
        sorted(
            component_id
            for component_id in expected_components
            if baseline.component_values[component_id] != mutated.component_values[component_id]
        )
    )
    authority = {
        "schema_version": "v7.2.reward-mutation-evidence.v1",
        "mutation_id": mutation_id,
        "mutation_implementation_digest": implementation_digest,
        "affected_component_id": component.component_id,
        "task_id": baseline.task_id,
        "reward_contract_id": contract.contract_id,
        "baseline_reward_record_id": baseline.reward_record_id,
        "mutated_reward_record_id": mutated.reward_record_id,
        "baseline_output_sha256": execution.baseline_output_sha256,
        "mutated_output_sha256": execution.mutated_output_sha256,
        "baseline_component_value": baseline_value,
        "mutated_component_value": mutated_value,
        "changed_component_ids": list(changed_component_ids),
        "allowed_collateral_component_ids": list(component.allowed_collateral_component_ids),
        "execution_evidence_ids": list(execution.execution_evidence_ids),
        "independently_recomputed": True,
    }
    return RewardMutationEvidence.model_validate_json(
        canonical_json(
            {
                "evidence_id": content_id("v72-reward-mutation-evidence", authority),
                **authority,
            }
        )
    )


def qualify_reward_causality(
    contract: RewardContract,
    registry: RewardMutationRegistry,
) -> RewardCausalityEvidence:
    """Execute every declared mutation and require component-specific reward loss."""
    evidence: list[RewardMutationEvidence] = []
    for component in contract.components:
        if not component.causal_mutation_ids:
            raise ValueError(f"reward component lacks causal mutations: {component.component_id}")
        for mutation_id in component.causal_mutation_ids:
            implementation_digest, hook = registry.resolve(mutation_id)
            evidence.append(
                _validate_execution(
                    contract,
                    component,
                    mutation_id,
                    implementation_digest,
                    hook(component),
                )
            )
    ordered = tuple(sorted(evidence, key=lambda item: item.evidence_id))
    authority = {
        "schema_version": "v7.2.reward-causality-evidence.v1",
        "reward_contract_id": contract.contract_id,
        "mutation_evidence": [item.model_dump(mode="json") for item in ordered],
        "qualified": True,
    }
    return RewardCausalityEvidence.model_validate_json(
        canonical_json(
            {
                "evidence_id": content_id("v72-reward-causality-evidence", authority),
                **authority,
            }
        )
    )


__all__ = [
    "RewardCausalityEvidence",
    "RewardMutationEvidence",
    "RewardMutationExecution",
    "RewardMutationRegistry",
    "qualify_reward_causality",
]
