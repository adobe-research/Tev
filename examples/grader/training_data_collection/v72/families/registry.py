"""Content-bound registry for complete v7.2 family implementations."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator
from typing import Annotated, Literal, Self, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator

from training_data_collection.common.identity import canonical_json, content_id
from training_data_collection.v72.contracts.identity import ContentIdentifier, StrictFrozenModel
from training_data_collection.v72.families.base import FamilyImplementation

NonEmptyString = Annotated[str, Field(min_length=1)]
_FAMILY_REVISION = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)+\.v[1-9][0-9]*$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_REQUIRED_METHODS = (
    "build_world",
    "build_workflow",
    "render_artifacts",
    "apply_corruptions",
    "build_public_specification",
    "build_hidden_suite",
)


class FamilyRegistration(StrictFrozenModel):
    """One exact logical revision and its lazily loaded implementation singleton."""

    schema_version: Literal["v7.2.family-registration.v1"] = "v7.2.family-registration.v1"
    registration_id: ContentIdentifier
    family_revision_id: NonEmptyString
    family_package: NonEmptyString
    implementation_entrypoint: NonEmptyString

    @field_validator("family_revision_id")
    @classmethod
    def validate_family_revision_id(cls, value: str) -> str:
        if _FAMILY_REVISION.fullmatch(value) is None:
            raise ValueError("family revision ID is not canonical")
        return value

    @field_validator("implementation_entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        if _ENTRYPOINT.fullmatch(value) is None:
            raise ValueError("family implementation entrypoint is invalid")
        return value

    @model_validator(mode="after")
    def validate_registration(self, info: ValidationInfo) -> Self:
        module_name = self.implementation_entrypoint.split(":", maxsplit=1)[0]
        if module_name != f"{self.family_package}.definition":
            raise ValueError("family entrypoint must resolve from its exact definition module")
        if not (info.context or {}).get("skip_identity") and (
            self.registration_id != self.expected_registration_id()
        ):
            raise ValueError("family registration identity is stale")
        return self

    def expected_registration_id(self) -> str:
        return content_id(
            "v72-family-registration",
            self.model_dump(mode="json", exclude={"registration_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> Self:
        provisional = cls.model_validate(
            {"registration_id": f"v72-family-registration:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            registration_id=provisional.expected_registration_id(),
            **provisional.model_dump(mode="python", exclude={"registration_id"}),
        )


class FamilyRegistryManifest(StrictFrozenModel):
    """Canonical registry manifest with no alias or fallback dispatch."""

    schema_version: Literal["v7.2.family-registry.v1"] = "v7.2.family-registry.v1"
    registry_id: ContentIdentifier
    registrations: tuple[FamilyRegistration, ...]

    @model_validator(mode="after")
    def validate_registry(self, info: ValidationInfo) -> Self:
        revisions = tuple(item.family_revision_id for item in self.registrations)
        packages = tuple(item.family_package for item in self.registrations)
        registration_ids = tuple(item.registration_id for item in self.registrations)
        if revisions != tuple(sorted(set(revisions))):
            raise ValueError("family revisions must be sorted and unique")
        if len(set(packages)) != len(packages):
            raise ValueError("family packages must be unique")
        if len(set(registration_ids)) != len(registration_ids):
            raise ValueError("family registration IDs must be unique")
        if not (info.context or {}).get("skip_identity") and (
            self.registry_id != self.expected_registry_id()
        ):
            raise ValueError("family registry identity is stale")
        return self

    def expected_registry_id(self) -> str:
        return content_id(
            "v72-family-registry",
            self.model_dump(mode="json", exclude={"registry_id"}),
        )

    @classmethod
    def identified(cls, registrations: tuple[FamilyRegistration, ...]) -> Self:
        provisional = cls.model_validate(
            {
                "registry_id": f"v72-family-registry:{'0' * 64}",
                "registrations": registrations,
            },
            context={"skip_identity": True},
        )
        return cls(
            registry_id=provisional.expected_registry_id(),
            registrations=provisional.registrations,
        )


def _registration(family_revision_id: str, family_package: str, symbol: str) -> FamilyRegistration:
    return FamilyRegistration.identified(
        family_revision_id=family_revision_id,
        family_package=family_package,
        implementation_entrypoint=f"{family_package}.definition:{symbol}",
    )


DEFAULT_FAMILY_REGISTRY = FamilyRegistryManifest.identified(
    tuple(
        sorted(
            (
                _registration(
                    "code.adversarial-program-tests-tb3.v1",
                    "training_data_collection.v72.families.code.adversarial_program_tests_tb3",
                    "ADVERSARIAL_PROGRAM_TESTS_TB3_FAMILY",
                ),
                _registration(
                    "code.adversarial-program-tests-tb4.v1",
                    "training_data_collection.v72.families.code.adversarial_program_tests_tb4",
                    "ADVERSARIAL_PROGRAM_TESTS_TB4_FAMILY",
                ),
                _registration(
                    "code.adversarial-program-tests.v2",
                    "training_data_collection.v72.families.code.adversarial_program_tests",
                    "ADVERSARIAL_PROGRAM_TESTS_FAMILY",
                ),
                _registration(
                    "code.elf-configuration-extraction-tb3.v1",
                    "training_data_collection.v72.families.code.elf_configuration_extraction_tb3",
                    "ELF_CONFIGURATION_EXTRACTION_TB3_FAMILY",
                ),
                _registration(
                    "code.elf-configuration-extraction-tb3.v2",
                    "training_data_collection.v72.families.code.elf_configuration_extraction_tb3_v2",
                    "ELF_CONFIGURATION_EXTRACTION_TB3_V2_FAMILY",
                ),
                _registration(
                    "code.elf-configuration-extraction-tb4.v1",
                    "training_data_collection.v72.families.code.elf_configuration_extraction_tb4",
                    "ELF_CONFIGURATION_EXTRACTION_TB4_FAMILY",
                ),
                _registration(
                    "code.elf-configuration-extraction.v2",
                    "training_data_collection.v72.families.code.elf_configuration_extraction",
                    "ELF_CONFIGURATION_EXTRACTION_FAMILY",
                ),
                _registration(
                    "code.file-format-carving-recovery.v1",
                    "training_data_collection.v72.families.code.file_format_carving_recovery",
                    "FILE_FORMAT_CARVING_RECOVERY_FAMILY",
                ),
                _registration(
                    "code.network-intrusion-triage-tb3.v1",
                    "training_data_collection.v72.families.code.network_intrusion_triage_tb3",
                    "NETWORK_INTRUSION_TRIAGE_TB3_FAMILY",
                ),
                _registration(
                    "code.network-intrusion-triage-tb4.v1",
                    "training_data_collection.v72.families.code.network_intrusion_triage_tb4",
                    "NETWORK_INTRUSION_TRIAGE_TB4_FAMILY",
                ),
                _registration(
                    "code.network-intrusion-triage.v2",
                    "training_data_collection.v72.families.code.network_intrusion_triage",
                    "NETWORK_INTRUSION_TRIAGE_FAMILY",
                ),
                _registration(
                    "code.registration-crackme.v1",
                    "training_data_collection.v72.families.code.registration_crackme",
                    "REGISTRATION_CRACKME_FAMILY",
                ),
                _registration(
                    "code.time-dependent-keygen.v1",
                    "training_data_collection.v72.families.code.time_dependent_keygen",
                    "TIME_DEPENDENT_KEYGEN_FAMILY",
                ),
                _registration(
                    "code.toy-isa-emulator-conformance.v1",
                    "training_data_collection.v72.families.code.toy_isa_emulator_conformance",
                    "TOY_ISA_EMULATOR_CONFORMANCE_FAMILY",
                ),
                _registration(
                    "code.legacy-batch-program-modernization.v1",
                    "training_data_collection.v72.families.code.legacy_batch_program_modernization",
                    "LEGACY_BATCH_PROGRAM_MODERNIZATION_FAMILY",
                ),
                _registration(
                    "cm.checkpoint-consolidation.v2",
                    "training_data_collection.v72.families.cm.checkpoint_consolidation",
                    "CHECKPOINT_CONSOLIDATION_FAMILY",
                ),
                _registration(
                    "cm.checkpoint-consolidation-tb3.v1",
                    "training_data_collection.v72.families.cm.checkpoint_consolidation_tb3",
                    "CHECKPOINT_CONSOLIDATION_TB3_FAMILY",
                ),
                _registration(
                    "cm.checkpoint-consolidation-tb3.v2",
                    "training_data_collection.v72.families.cm.checkpoint_consolidation_tb3_v2",
                    "CHECKPOINT_CONSOLIDATION_TB3_V2_FAMILY",
                ),
                _registration(
                    "cm.checkpoint-consolidation-tb4.v1",
                    "training_data_collection.v72.families.cm.checkpoint_consolidation_tb4",
                    "CHECKPOINT_CONSOLIDATION_TB4_FAMILY",
                ),
                _registration(
                    "cm.intrastat-trade-filing.v1",
                    "training_data_collection.v72.families.cm.intrastat_trade_filing",
                    "INTRASTAT_TRADE_FILING_FAMILY",
                ),
                _registration(
                    "cm.ranking-workspace-recovery-tb3.v1",
                    "training_data_collection.v72.families.cm.ranking_workspace_recovery_tb3",
                    "RANKING_WORKSPACE_RECOVERY_TB3_FAMILY",
                ),
                _registration(
                    "cm.ranking-workspace-recovery-tb4.v1",
                    "training_data_collection.v72.families.cm.ranking_workspace_recovery_tb4",
                    "RANKING_WORKSPACE_RECOVERY_TB4_FAMILY",
                ),
                _registration(
                    "cm.ranking-workspace-recovery.v2",
                    "training_data_collection.v72.families.cm.ranking_recovery",
                    "RANKING_RECOVERY_FAMILY",
                ),
                _registration(
                    "cm.rules-driven-claims-adjudication.v1",
                    "training_data_collection.v72.families.cm.rules_driven_claims_adjudication",
                    "RULES_DRIVEN_CLAIMS_ADJUDICATION_FAMILY",
                ),
                _registration(
                    "cm.warehouse-etl-tb3.v1",
                    "training_data_collection.v72.families.cm.warehouse_etl_tb3",
                    "WAREHOUSE_ETL_TB3_FAMILY",
                ),
                _registration(
                    "cm.warehouse-etl-tb4.v1",
                    "training_data_collection.v72.families.cm.warehouse_etl_tb4",
                    "WAREHOUSE_ETL_TB4_FAMILY",
                ),
                _registration(
                    "cm.warehouse-etl.v2",
                    "training_data_collection.v72.families.cm.warehouse_etl_recovery",
                    "WAREHOUSE_ETL_RECOVERY_FAMILY",
                ),
                _registration(
                    "ml.causal-structure-inference.v1",
                    "training_data_collection.v72.families.ml.causal_structure_inference",
                    "CAUSAL_STRUCTURE_INFERENCE_FAMILY",
                ),
                _registration(
                    "ml.counterfactual-recommender-tb3.v1",
                    "training_data_collection.v72.families.ml.counterfactual_recommender_tb3",
                    "COUNTERFACTUAL_RECOMMENDER_TB3_FAMILY",
                ),
                _registration(
                    "ml.counterfactual-recommender-tb4.v1",
                    "training_data_collection.v72.families.ml.counterfactual_recommender_tb4",
                    "COUNTERFACTUAL_RECOMMENDER_TB4_FAMILY",
                ),
                _registration(
                    "ml.counterfactual-recommender.v2",
                    "training_data_collection.v72.families.ml.counterfactual_recommender",
                    "COUNTERFACTUAL_RECOMMENDER_FAMILY",
                ),
                _registration(
                    "ml.embedding-drift-monitoring.v1",
                    "training_data_collection.v72.families.ml.embedding_drift_monitoring",
                    "EMBEDDING_DRIFT_MONITORING_FAMILY",
                ),
                _registration(
                    "ml.paper-result-reproduction.v1",
                    "training_data_collection.v72.families.ml.paper_result_reproduction",
                    "PAPER_RESULT_REPRODUCTION_FAMILY",
                ),
                _registration(
                    "mathematical.abelian-extension-classification.v1",
                    "training_data_collection.v72.families.mathematical.abelian_extension_classification",
                    "ABELIAN_EXTENSION_CLASSIFICATION_FAMILY",
                ),
                _registration(
                    "mathematical.sound-change-rule-induction.v1",
                    "training_data_collection.v72.families.mathematical.sound_change_rule_induction",
                    "SOUND_CHANGE_RULE_INDUCTION_FAMILY",
                ),
                _registration(
                    "ml.repository-repair-tb3.v1",
                    "training_data_collection.v72.families.ml.repository_repair_tb3",
                    "REPOSITORY_REPAIR_TB3_FAMILY",
                ),
                _registration(
                    "ml.repository-repair-tb3.v2",
                    "training_data_collection.v72.families.ml.repository_repair_tb3_v2",
                    "REPOSITORY_REPAIR_TB3_V2_FAMILY",
                ),
                _registration(
                    "ml.repository-repair-tb4.v1",
                    "training_data_collection.v72.families.ml.repository_repair_tb4",
                    "REPOSITORY_REPAIR_TB4_FAMILY",
                ),
                _registration(
                    "ml.repository-repair.v2",
                    "training_data_collection.v72.families.ml.repository_repair",
                    "REPOSITORY_REPAIR_FAMILY",
                ),
                _registration(
                    "optimization.calendar-constraint-scheduling.v1",
                    "training_data_collection.v72.families.optimization.calendar_constraint_scheduling",
                    "CALENDAR_CONSTRAINT_SCHEDULING_FAMILY",
                ),
                _registration(
                    "optimization.constrained-solver.v2",
                    "training_data_collection.v72.families.optimization.constrained_solver",
                    "CONSTRAINED_SOLVER_FAMILY",
                ),
                _registration(
                    "optimization.constrained-solver-tb3.v1",
                    "training_data_collection.v72.families.optimization.constrained_solver_tb3",
                    "CONSTRAINED_SOLVER_TB3_FAMILY",
                ),
                _registration(
                    "optimization.constrained-solver-tb3.v2",
                    "training_data_collection.v72.families.optimization.constrained_solver_tb3_v2",
                    "CONSTRAINED_SOLVER_TB3_V2_FAMILY",
                ),
                _registration(
                    "optimization.constrained-solver-tb4.v1",
                    "training_data_collection.v72.families.optimization.constrained_solver_tb4",
                    "CONSTRAINED_SOLVER_TB4_FAMILY",
                ),
                _registration(
                    "optimization.game-theory-equilibrium.v1",
                    "training_data_collection.v72.families.optimization.game_theory_equilibrium",
                    "GAME_THEORY_EQUILIBRIUM_FAMILY",
                ),
                _registration(
                    "optimization.inference-batching-schedule.v1",
                    "training_data_collection.v72.families.optimization.inference_batching_schedule",
                    "INFERENCE_BATCHING_SCHEDULE_FAMILY",
                ),
                _registration(
                    "physical.ising-measurement-pipeline.v2",
                    "training_data_collection.v72.families.physical.ising_measurement_pipeline",
                    "ISING_MEASUREMENT_PIPELINE_FAMILY",
                ),
                _registration(
                    "physical.nonlinear-state-estimation.v2",
                    "training_data_collection.v72.families.physical.nonlinear_state_estimation",
                    "NONLINEAR_STATE_ESTIMATION_FAMILY",
                ),
                _registration(
                    "physical.qec-circuit-decoder.v2",
                    "training_data_collection.v72.families.physical.qec_circuit_decoder",
                    "QEC_CIRCUIT_DECODER_FAMILY",
                ),
                _registration(
                    "sre.cloud-cost-remediation.v2",
                    "training_data_collection.v72.families.sre.cloud_cost_remediation",
                    "CLOUD_COST_REMEDIATION_FAMILY",
                ),
                _registration(
                    "sre.event-log-exactly-once-replay.v1",
                    "training_data_collection.v72.families.sre.event_log_exactly_once_replay",
                    "EVENT_LOG_EXACTLY_ONCE_REPLAY_FAMILY",
                ),
                _registration(
                    "sre.cloud-cost-remediation-tb3.v1",
                    "training_data_collection.v72.families.sre.cloud_cost_remediation_tb3",
                    "CLOUD_COST_REMEDIATION_TB3_FAMILY",
                ),
                _registration(
                    "sre.cloud-cost-remediation-tb4.v1",
                    "training_data_collection.v72.families.sre.cloud_cost_remediation_tb4",
                    "CLOUD_COST_REMEDIATION_TB4_FAMILY",
                ),
                _registration(
                    "sre.kubernetes-incident-analysis.v1",
                    "training_data_collection.v72.families.sre.kubernetes_incident_analysis",
                    "KUBERNETES_INCIDENT_ANALYSIS_FAMILY",
                ),
                _registration(
                    "sre.kubernetes-migration.v1",
                    "training_data_collection.v72.families.sre.kubernetes_migration",
                    "KUBERNETES_MIGRATION_FAMILY",
                ),
                _registration(
                    "sre.log-permission-remediation.v1",
                    "training_data_collection.v72.families.sre.log_permission_remediation",
                    "LOG_PERMISSION_REMEDIATION_FAMILY",
                ),
                _registration(
                    "sre.package-dependency-lock-resolution.v1",
                    "training_data_collection.v72.families.sre.package_dependency_lock_resolution",
                    "PACKAGE_DEPENDENCY_LOCK_RESOLUTION_FAMILY",
                ),
            ),
            key=lambda item: item.family_revision_id,
        )
    )
)


class FamilyRegistry:
    """Resolve only exact entries from a validated immutable manifest."""

    def __init__(self, manifest: FamilyRegistryManifest = DEFAULT_FAMILY_REGISTRY) -> None:
        self._manifest = FamilyRegistryManifest.model_validate_json(
            canonical_json(manifest.model_dump(mode="json"))
        )
        self._by_revision = {item.family_revision_id: item for item in self._manifest.registrations}

    @property
    def manifest(self) -> FamilyRegistryManifest:
        return self._manifest

    def __iter__(self) -> Iterator[FamilyRegistration]:
        return iter(self._manifest.registrations)

    def registration(self, family_revision_id: str) -> FamilyRegistration:
        try:
            return self._by_revision[family_revision_id]
        except KeyError as error:
            raise KeyError(f"unregistered v7.2 family revision: {family_revision_id}") from error

    def resolve(self, family_revision_id: str) -> FamilyImplementation:
        registration = self.registration(family_revision_id)
        module_name, qualname = registration.implementation_entrypoint.split(":", maxsplit=1)
        target: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        if any(not callable(getattr(target, method, None)) for method in _REQUIRED_METHODS):
            raise TypeError("registered family does not implement the complete family protocol")
        family_entrypoint = getattr(target, "family_implementation_entrypoint", None)
        expected_module = f"{registration.family_package}.definition"
        if family_entrypoint is None:
            if type(target).__module__ != expected_module:
                raise ValueError("registered family implementation authority leaves its package")
        elif type(family_entrypoint) is not str or not family_entrypoint.startswith(
            f"{registration.family_package}."
        ):
            raise ValueError("registered family implementation authority leaves its package")
        return cast("FamilyImplementation", target)


__all__ = [
    "DEFAULT_FAMILY_REGISTRY",
    "FamilyRegistration",
    "FamilyRegistry",
    "FamilyRegistryManifest",
]
