"""Independently recomputable reward contracts and result records."""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from training_data_collection.common.identity import content_id

from .identity import ContentIdentifier, StrictFrozenModel


class RewardComponent(StrictFrozenModel):
    """Public and executable authority for one partial-credit component."""

    component_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(gt=0.0, le=1.0)
    public_rule_ids: tuple[str, ...] = Field(min_length=1)
    workflow_stage_ids: tuple[str, ...] = Field(min_length=1)
    checker_implementation_id: ContentIdentifier
    causal_mutation_ids: tuple[str, ...] = Field(min_length=1)
    allowed_collateral_component_ids: tuple[str, ...] = ()
    maximum_reward: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_component(self) -> Self:
        for name, values in (
            ("public rules", self.public_rule_ids),
            ("workflow stages", self.workflow_stage_ids),
            ("causal mutations", self.causal_mutation_ids),
            ("allowed collateral components", self.allowed_collateral_component_ids),
        ):
            if any(not value.strip() for value in values) or len(values) != len(set(values)):
                raise ValueError(f"reward component {name} must be nonblank and unique")
        if not math.isfinite(self.weight) or not math.isfinite(self.maximum_reward):
            raise ValueError("reward component bounds must be finite")
        if self.component_id in self.allowed_collateral_component_ids:
            raise ValueError("a reward component cannot declare itself as collateral")
        return self


class RewardContract(StrictFrozenModel):
    """Content-bound vector reward authority for one family revision."""

    schema_version: Literal["v7.2.reward.v1"] = "v7.2.reward.v1"
    contract_id: ContentIdentifier
    components: tuple[RewardComponent, ...] = Field(min_length=1)
    hard_gate_public_rule_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self, info: Any) -> Self:
        component_ids = [component.component_id for component in self.components]
        if component_ids != sorted(component_ids) or len(component_ids) != len(set(component_ids)):
            raise ValueError("reward components must be sorted by unique component identity")
        if any(not rule_id.strip() for rule_id in self.hard_gate_public_rule_ids) or len(
            self.hard_gate_public_rule_ids
        ) != len(set(self.hard_gate_public_rule_ids)):
            raise ValueError("hard-gate public rules must be nonblank and unique")
        known_components = set(component_ids)
        for component in self.components:
            unknown = set(component.allowed_collateral_component_ids) - known_components
            if unknown:
                raise ValueError(
                    f"reward component collateral dependencies are unknown: {sorted(unknown)}"
                )
        total_weight = sum(component.weight for component in self.components)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-12):
            raise ValueError("reward component weights must sum to one")
        if not (info.context or {}).get("skip_identity") and (
            self.contract_id != self.expected_contract_id()
        ):
            raise ValueError("reward contract identity does not match its authority")
        return self

    def expected_contract_id(self) -> str:
        return content_id(
            "v72-reward-contract", self.model_dump(mode="json", exclude={"contract_id"})
        )

    @classmethod
    def identified(
        cls,
        *,
        components: tuple[RewardComponent, ...],
        hard_gate_public_rule_ids: tuple[str, ...],
    ) -> Self:
        values = {
            "components": tuple(sorted(components, key=lambda item: item.component_id)),
            "hard_gate_public_rule_ids": hard_gate_public_rule_ids,
        }
        provisional = cls.model_validate(
            {"contract_id": f"v72-reward-contract:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(contract_id=provisional.expected_contract_id(), **values)


RewardStatus = Literal["policy_result", "infrastructure_error", "task_defect", "verifier_error"]


class RewardResult(StrictFrozenModel):
    """Verifier result whose scalar reward is derived from its component vector."""

    schema_version: Literal["v7.2.reward-result.v1"] = "v7.2.reward-result.v1"
    reward_record_id: ContentIdentifier
    task_id: ContentIdentifier
    reward_contract_id: ContentIdentifier
    status: RewardStatus
    component_values: dict[str, float]
    hard_gate_values: dict[str, bool]
    total: float | None
    evidence_record_ids: tuple[ContentIdentifier, ...]
    failure_detail: str | None = None

    @model_validator(mode="after")
    def validate_result(self, info: Any) -> Self:
        if self.status == "policy_result":
            if not self.component_values or not self.hard_gate_values or self.total is None:
                raise ValueError("policy results require components, hard gates, and a total")
            values = tuple(self.component_values.values())
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("reward component values must be finite fractions")
            if not math.isfinite(self.total) or not 0.0 <= self.total <= 1.0:
                raise ValueError("reward total must be a finite fraction")
            if self.failure_detail is not None:
                raise ValueError("healthy policy results cannot carry system failure detail")
        elif self.component_values or self.hard_gate_values or self.total is not None:
            raise ValueError("non-policy outcomes cannot be encoded as reward-zero results")
        elif not self.failure_detail:
            raise ValueError("non-policy outcomes require a failure detail")
        if len(self.evidence_record_ids) != len(set(self.evidence_record_ids)):
            raise ValueError("reward evidence identities must be unique")
        if not (info.context or {}).get("skip_identity") and (
            self.reward_record_id != self.expected_reward_record_id()
        ):
            raise ValueError("reward result identity does not match its authority")
        return self

    def expected_reward_record_id(self) -> str:
        return content_id(
            "v72-reward-result",
            self.model_dump(mode="json", exclude={"reward_record_id"}),
        )

    @classmethod
    def from_policy_result(
        cls,
        *,
        task_id: str,
        contract: RewardContract,
        component_values: dict[str, float],
        hard_gate_values: dict[str, bool],
        evidence_record_ids: tuple[str, ...],
    ) -> Self:
        expected_components = {component.component_id for component in contract.components}
        if set(component_values) != expected_components:
            raise ValueError("reward result component set differs from its contract")
        if set(hard_gate_values) != set(contract.hard_gate_public_rule_ids):
            raise ValueError("reward result hard-gate set differs from its contract")
        total = sum(
            component.weight * component_values[component.component_id]
            for component in contract.components
        )
        if not all(hard_gate_values.values()):
            total = 0.0
        values = {
            "task_id": task_id,
            "reward_contract_id": contract.contract_id,
            "status": "policy_result",
            "component_values": component_values,
            "hard_gate_values": hard_gate_values,
            "total": total,
            "evidence_record_ids": evidence_record_ids,
        }
        provisional = cls.model_validate(
            {"reward_record_id": f"v72-reward-result:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(reward_record_id=provisional.expected_reward_record_id(), **values)

    @classmethod
    def from_system_failure(
        cls,
        *,
        task_id: str,
        reward_contract_id: str,
        status: Literal["infrastructure_error", "task_defect", "verifier_error"],
        failure_detail: str,
        evidence_record_ids: tuple[str, ...] = (),
    ) -> Self:
        values = {
            "task_id": task_id,
            "reward_contract_id": reward_contract_id,
            "status": status,
            "component_values": {},
            "hard_gate_values": {},
            "total": None,
            "evidence_record_ids": evidence_record_ids,
            "failure_detail": failure_detail,
        }
        provisional = cls.model_validate(
            {"reward_record_id": f"v72-reward-result:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(reward_record_id=provisional.expected_reward_record_id(), **values)
