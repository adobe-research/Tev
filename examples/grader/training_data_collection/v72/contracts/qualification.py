"""Strict policy contracts and content bindings for v7.2 qualification."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import Field, model_validator

from training_data_collection.common.identity import content_digest
from training_data_collection.v72.contracts.identity import StrictFrozenModel

NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class IntegerRange(StrictFrozenModel):
    minimum: NonNegativeInteger
    maximum: NonNegativeInteger

    @model_validator(mode="after")
    def validate_order(self) -> IntegerRange:
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class WorldPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.world-policy.v1"]
    policy_id: NonEmptyString
    required_seed_namespaces: tuple[
        Literal["public", "private-world", "hidden-case", "perturbation", "behavior"], ...
    ]
    deterministic_rebuild_count: PositiveInteger
    content_identified_recipe_required: bool
    content_identified_world_required: bool
    executable_invariant_validators_required: bool
    authority_input_identity_sensitivity_required: bool
    private_world_hidden_from_policy: bool
    undeclared_randomness_allowed: bool
    current_time_as_authority_allowed: bool
    host_path_as_authority_allowed: bool
    network_during_generation_allowed: bool
    rendered_files_may_be_latent_world_authority: bool


class ArtifactRelationPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.artifact-relation-policy.v1"]
    policy_id: NonEmptyString
    allowed_semantic_roles: tuple[
        Literal[
            "authoritative_source",
            "raw_observation_or_log",
            "lossy_projection",
            "stale_derived_state",
            "controlled_corrupted_artifact",
            "configuration",
            "executable_or_harness",
            "schema_or_interface_contract",
            "protected_state",
            "distractor",
            "intermediate_state",
            "output_template",
            "required_output",
            "evidence_artifact",
        ],
        ...,
    ]
    allowed_relation_types: tuple[
        Literal[
            "derived_from",
            "generated_from",
            "summarizes",
            "indexes",
            "shards",
            "reassembles",
            "joins_on",
            "aligns_with",
            "must_match",
            "hashes",
            "constrained_by",
            "contradicts",
            "supersedes",
            "must_preserve",
            "protected_by",
            "validated_by",
            "consumed_by",
            "produced_by",
        ],
        ...,
    ]
    realized_artifact_validation_required: bool
    baseline_and_corrupted_validation_required: bool
    critical_relation_validator_required: bool
    critical_public_artifact_causal_route_required: bool
    unknown_endpoints_allowed: bool
    unsafe_paths_allowed: bool
    unapproved_derivation_cycles_allowed: bool
    metadata_only_relation_evidence_allowed: bool


class WorkflowPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.workflow-policy.v1"]
    policy_id: NonEmptyString
    integration_critical_path: IntegerRange
    frontier_critical_path: IntegerRange
    minimum_verifier_observations_per_critical_stage: PositiveInteger
    acyclic_graph_required: bool
    endpoint_resolution_required: bool
    reachable_terminal_outputs_required: bool
    connected_critical_path_required: bool
    independently_observable_stage_state_required: bool
    expected_failure_transitions_required: bool
    self_report_may_establish_stage_execution: bool
    critical_stage_ablation_required: bool
    stale_intermediate_ablation_required: bool


class CorruptionPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.corruption-policy.v1"]
    policy_id: NonEmptyString
    allowed_operator_classes: tuple[
        Literal[
            "stale_snapshot",
            "dropped_record",
            "reordered_event",
            "schema_drift",
            "conflicting_metadata",
            "partial_write",
            "wrong_shard_map",
            "decoy_traffic",
            "hidden_topology_change",
            "forged_evidence",
            "interacting_configuration_defect",
        ],
        ...,
    ]
    render_consistent_baseline_first_required: bool
    baseline_projection_validation_required: bool
    registered_operators_required: bool
    declared_effect_observation_required: bool
    unaffected_relation_preservation_required: bool
    expected_metamorphic_effects_required: bool
    native_validity_preserved_unless_declared: bool
    undeclared_corruptions_allowed: bool


class HorizonBand(StrictFrozenModel):
    semantic_critical_path: IntegerRange
    expected_actions: IntegerRange
    operational_transitions: IntegerRange
    verification_checkpoints: IntegerRange


class HorizonPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.horizon-policy.v1"]
    policy_id: NonEmptyString
    primitive: HorizonBand
    integration: HorizonBand
    frontier: HorizonBand
    required_dimensions: tuple[Literal["semantic", "action", "operational", "verification"], ...]
    context_footprint_recorded_separately: bool
    resource_evidence_recorded_separately: bool
    bytes_as_semantic_authority: bool
    file_count_as_semantic_authority: bool
    prompt_length_as_semantic_authority: bool
    nominal_stage_count_as_semantic_authority: bool
    efficient_policy_must_pad_actions: bool
    evidence_record_ids_required: bool


class PublicContractPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.public-contract-policy.v1"]
    policy_id: NonEmptyString
    required_workspace_paths: tuple[Literal["task_spec.json", "prompt.md", "input", "output"], ...]
    supported_native_schema_kinds: tuple[
        Literal[
            "binary",
            "csv",
            "elf",
            "json",
            "npz",
            "pcap",
            "python_executable",
            "qasm",
            "safetensors",
            "sqlite",
            "text",
            "yaml",
        ],
        ...,
    ]
    canonical_prompt_required: bool
    complete_public_input_coverage_required: bool
    executable_public_validators_required: bool
    exact_requirement_references_required: bool
    exact_output_paths_required: bool
    native_output_schemas_required: bool
    authority_and_precedence_rules_required: bool
    positive_and_targeted_negative_validation_required: bool
    private_undocumented_obligations_allowed: bool
    generic_unrelated_input_output_bindings_allowed: bool


class IndependencePolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.independence-policy.v1"]
    policy_id: NonEmptyString
    required_authorities: tuple[
        Literal["materializer", "reference", "alternative", "verifier"], ...
    ]
    allowed_shared_module_categories: tuple[
        Literal["contracts", "canonical_serialization", "hashing", "reviewed_native_io"],
        ...,
    ]
    source_tree_digests_required: bool
    dependency_closure_digests_required: bool
    algorithm_description_digests_required: bool
    separate_process_execution_required: bool
    public_only_alternative_required: bool
    full_reward_for_both_valid_solvers_required: bool
    distinct_output_digest_required: bool
    semantic_cross_authority_imports_allowed: bool
    shared_oracle_functions_allowed: bool
    shared_expected_value_tables_allowed: bool
    witness_byte_comparison_allowed: bool


class ExploitPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.exploit-policy.v1"]
    policy_id: NonEmptyString
    required_probe_classes: tuple[
        Literal[
            "empty_output",
            "minimal_schema_valid",
            "constant_output",
            "copied_input",
            "one_seed_hard_coding",
            "family_template_hard_coding",
            "ignore_critical_role",
            "ignore_advertised_signal",
            "evidence_omission",
            "contradictory_report",
            "visible_only_overfit",
            "sidecar_omission",
            "path_traversal",
            "private_lookup",
        ],
        ...,
    ]
    obvious_shortcut_maximum_reward: Fraction
    run_every_probe_on_every_representative_cell: bool
    policy_boundary_execution_required: bool
    strategy_configuration_required: bool
    read_write_trace_required: bool
    exit_status_required: bool
    sealed_output_digest_required: bool
    component_reward_evidence_required: bool
    component_specific_ceilings_required: bool
    distinct_strategy_implementation_digests_required: bool
    traversal_and_private_lookup_attempts_must_execute: bool
    stale_output_must_lose_affected_component: bool


class RewardPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.reward-policy.v1"]
    policy_id: NonEmptyString
    scalar_minimum: float
    scalar_maximum: float
    recommended_component_kinds: tuple[
        Literal[
            "structural_validity",
            "intermediate_reconstruction_correctness",
            "terminal_native_artifact_correctness",
            "cross_artifact_consistency",
            "preservation_and_safety",
            "evidence_quality",
            "hidden_case_generalization",
        ],
        ...,
    ]
    independently_recomputed_components_required: bool
    public_rule_and_stage_mapping_required: bool
    independent_checker_required_per_component: bool
    causal_mutation_test_required_per_component: bool
    stage_reward_events_required: bool
    structural_validation_may_establish_semantic_correctness: bool
    self_reported_values_may_establish_correctness: bool
    evidence_presence_alone_may_earn_credit: bool
    private_hard_gates_may_add_obligations: bool
    policy_result_is_only_reward_status: bool

    @model_validator(mode="after")
    def validate_scalar_range(self) -> RewardPolicy:
        if self.scalar_minimum >= self.scalar_maximum:
            raise ValueError("scalar_minimum must be less than scalar_maximum")
        return self


class SourcePolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.source-policy.v1"]
    policy_id: NonEmptyString
    allowed_source_kinds: tuple[
        Literal[
            "generated",
            "repository",
            "dataset",
            "paper",
            "packet_capture",
            "compiled_binary",
            "checkpoint",
            "live_system",
        ],
        ...,
    ]
    required_authority_fields: tuple[
        Literal[
            "origin",
            "immutable_version",
            "license",
            "acquisition_record",
            "transformation_chain",
            "source_tree_sha256",
        ],
        ...,
    ]
    source_level_split_before_rendering_required: bool
    immutable_source_version_required: bool
    license_evidence_required: bool
    transformation_lineage_required: bool
    decontamination_required: bool
    network_during_package_generation_allowed: bool
    network_during_verification_allowed: bool
    generated_only_frontier_claim_allowed: bool
    restricted_assets_may_be_published: bool


class BenchmarkAnchorPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.benchmark-anchor-policy.v1"]
    policy_id: NonEmptyString
    exact_family_revision_binding_required: bool
    anchor_public_metadata_digest_required: bool
    automated_evidence_binding_gate_required: bool
    permitted_decisions: tuple[Literal["comparable", "not_comparable"], ...]
    not_comparable_blocks_benchmark_quality_release: bool
    benchmark_prompt_as_training_asset_allowed: bool
    benchmark_input_as_training_asset_allowed: bool
    benchmark_answer_as_training_asset_allowed: bool
    benchmark_grader_as_training_asset_allowed: bool
    review_may_waive_executable_gate: bool


class CoveragePolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.coverage-policy.v1"]
    policy_id: NonEmptyString
    required_subdomains: tuple[Literal["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7"], ...]
    forbidden_subdomains: tuple[Literal["14.1"], ...]
    required_axes: tuple[
        Literal[
            "curriculum_layer",
            "subdomain",
            "family_revision",
            "scale",
            "semantic_horizon",
            "action_horizon",
            "operational_horizon",
            "verification_horizon",
            "workflow_stage",
            "tool_capability",
            "artifact_role",
            "relation_type",
            "corruption_class",
            "source_kind",
            "license_class",
            "reward_component",
            "observed_reward_band",
            "outcome_category",
            "failure_category",
        ],
        ...,
    ]
    minimum_covering_array_strength: PositiveInteger
    minimum_deterministic_seeds_per_required_cell: PositiveInteger
    minimum_families_per_subdomain: PositiveInteger
    maximum_family_fraction: Fraction
    maximum_duplicate_public_tree_fraction: Fraction
    every_axis_value_required: bool
    every_required_pair_required: bool
    caller_asserted_coverage_allowed: bool


class CurriculumPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.curriculum-policy.v1"]
    policy_id: NonEmptyString
    required_layers: tuple[
        Literal["primitive_terminal", "family_integration", "professional_frontier"], ...
    ]
    minimum_tasks_per_required_layer: PositiveInteger
    exactly_one_layer_per_task_required: bool
    executable_layer_evidence_required: bool
    integration_multi_artifact_causality_required: bool
    frontier_realistic_tooling_required: bool
    frontier_governed_source_or_real_system_required: bool
    reward_band_alone_may_define_curriculum: bool
    sampler_layer_targets_required: bool
    sampler_horizon_targets_required: bool
    sampler_source_targets_required: bool
    sampler_family_targets_required: bool
    sampler_reward_band_targets_required: bool


REQUIRED_PRODUCTION_COVERAGE_AXES = (
    "curriculum_layer",
    "subdomain",
    "family_revision",
    "scale",
    "semantic_horizon",
    "action_horizon",
    "operational_horizon",
    "verification_horizon",
    "workflow_stage",
    "tool_capability",
    "artifact_role",
    "relation_type",
    "corruption_class",
    "source_kind",
    "license_class",
    "reward_component",
    "observed_reward_band",
    "outcome_category",
    "failure_category",
)
REQUIRED_PRODUCTION_SUBDOMAINS = ("9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7")
REQUIRED_PRODUCTION_CURRICULUM_LAYERS = (
    "primitive_terminal",
    "family_integration",
    "professional_frontier",
)


def require_production_coverage_policy(policy: CoveragePolicy) -> None:
    """Reject any semantic weakening of the corpus-wide production coverage gate."""
    if (
        policy.policy_id != "v7.2-multidimensional-coverage-v1"
        or policy.required_subdomains != REQUIRED_PRODUCTION_SUBDOMAINS
        or policy.forbidden_subdomains != ("14.1",)
        or policy.required_axes != REQUIRED_PRODUCTION_COVERAGE_AXES
        or policy.minimum_covering_array_strength < 2
        or policy.minimum_deterministic_seeds_per_required_cell < 3
        or policy.minimum_families_per_subdomain < 2
        or policy.maximum_family_fraction > 0.1
        or policy.maximum_duplicate_public_tree_fraction != 0.0
        or not policy.every_axis_value_required
        or not policy.every_required_pair_required
        or policy.caller_asserted_coverage_allowed
    ):
        raise ValueError("coverage policy weakens mandatory production invariants")


def require_production_curriculum_policy(policy: CurriculumPolicy) -> None:
    """Reject any semantic weakening of the exact three-layer curriculum gate."""
    if (
        policy.policy_id != "v7.2-three-layer-curriculum-v1"
        or policy.required_layers != REQUIRED_PRODUCTION_CURRICULUM_LAYERS
        or policy.minimum_tasks_per_required_layer < 1
        or not policy.exactly_one_layer_per_task_required
        or not policy.executable_layer_evidence_required
        or not policy.integration_multi_artifact_causality_required
        or not policy.frontier_realistic_tooling_required
        or not policy.frontier_governed_source_or_real_system_required
        or policy.reward_band_alone_may_define_curriculum
        or not policy.sampler_layer_targets_required
        or not policy.sampler_horizon_targets_required
        or not policy.sampler_source_targets_required
        or not policy.sampler_family_targets_required
        or not policy.sampler_reward_band_targets_required
    ):
        raise ValueError("curriculum policy weakens mandatory production invariants")


class RolloutPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.rollout-policy.v1"]
    policy_id: NonEmptyString
    maximum_action_count: PositiveInteger
    maximum_tool_call_count: PositiveInteger
    maximum_input_tokens: PositiveInteger
    maximum_output_tokens: PositiveInteger
    maximum_elapsed_seconds: PositiveInteger
    candidate_workspace_and_verifier_required: bool
    production_execution_profile_required: bool
    public_private_workspace_isolation_required: bool
    sealed_output_before_verifier_required: bool
    action_and_tool_trace_required: bool
    token_and_elapsed_metrics_required: bool
    stage_reward_events_required: bool
    termination_cause_required: bool
    execution_profile_cost_required: bool
    policy_infrastructure_task_verifier_status_separation_required: bool
    healthy_runtime_required_for_policy_timeout_classification: bool


class BehavioralPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.behavioral-policy.v1"]
    policy_id: NonEmptyString
    minimum_policy_outcomes_per_task: PositiveInteger
    minimum_policy_outcomes_per_family_scale: PositiveInteger
    minimum_rollout_seeds_per_task: PositiveInteger
    maximum_family_scale_full_reward_fraction: Fraction
    maximum_family_scale_zero_reward_fraction: Fraction
    fixed_policy_identity_required: bool
    stage_completion_evidence_required: bool
    tool_use_evidence_required: bool
    horizon_evidence_required: bool
    failure_taxonomy_required: bool
    unobserved_task_allowed: bool
    insufficient_task_allowed: bool
    unresolved_task_error_allowed: bool
    unresolved_contract_error_allowed: bool
    unresolved_verifier_error_allowed: bool
    unresolved_infrastructure_error_allowed: bool
    infrastructure_errors_are_policy_rewards: bool
    task_defects_are_policy_rewards: bool


class ReleasePolicyBindings(StrictFrozenModel):
    world: NonEmptyString
    artifact_relation: NonEmptyString
    workflow: NonEmptyString
    corruption: NonEmptyString
    horizon: NonEmptyString
    public_contract: NonEmptyString
    independence: NonEmptyString
    exploit: NonEmptyString
    reward: NonEmptyString
    source: NonEmptyString
    benchmark_anchor: NonEmptyString
    coverage: NonEmptyString
    curriculum: NonEmptyString
    rollout: NonEmptyString
    behavioral: NonEmptyString
    release: NonEmptyString
    export: NonEmptyString


type PolicyName = Literal[
    "world",
    "artifact_relation",
    "workflow",
    "corruption",
    "horizon",
    "public_contract",
    "independence",
    "exploit",
    "reward",
    "source",
    "benchmark_anchor",
    "coverage",
    "curriculum",
    "rollout",
    "behavioral",
    "release",
    "export",
]


class InvariantPolicyBindings(StrictFrozenModel):
    v72_world_001: tuple[PolicyName, ...]
    v72_graph_001: tuple[PolicyName, ...]
    v72_flow_001: tuple[PolicyName, ...]
    v72_corrupt_001: tuple[PolicyName, ...]
    v72_public_001: tuple[PolicyName, ...]
    v72_verify_001: tuple[PolicyName, ...]
    v72_indep_001: tuple[PolicyName, ...]
    v72_exploit_001: tuple[PolicyName, ...]
    v72_horizon_001: tuple[PolicyName, ...]
    v72_source_001: tuple[PolicyName, ...]
    v72_layer_001: tuple[PolicyName, ...]
    v72_runtime_001: tuple[PolicyName, ...]
    v72_rl_001: tuple[PolicyName, ...]
    v72_anchor_001: tuple[PolicyName, ...]
    v72_release_001: tuple[PolicyName, ...]
    v72_export_001: tuple[PolicyName, ...]

    @model_validator(mode="after")
    def validate_nonempty_bindings(self) -> InvariantPolicyBindings:
        if any(not names for names in self.__dict__.values()):
            raise ValueError("every invariant must bind at least one policy")
        return self


class ReleasePolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.release-policy.v1"]
    policy_id: NonEmptyString
    product: Literal["immutable_observed_online_rl_task_release"]
    policy_bindings: ReleasePolicyBindings
    invariant_policy_bindings: InvariantPolicyBindings
    required_evidence_kinds: tuple[
        Literal[
            "world_generation",
            "artifact_relation_execution",
            "workflow_causality",
            "corruption_locality",
            "public_contract",
            "implementation_independence",
            "hidden_and_metamorphic_coverage",
            "exploit_execution",
            "horizon_and_curriculum",
            "source_and_split_governance",
            "benchmark_anchor_review",
            "production_runtime_parity",
            "reward_causality",
            "instance_admission",
            "release_candidate",
            "behavioral_qualification",
            "coverage",
        ],
        ...,
    ]
    accepted_task_schema_versions: tuple[Literal["v7.2.task.v1"], ...]
    schema_revision_inferred_from_every_task_required: bool
    semantic_policy_parsing_required: bool
    evidence_identity_recomputation_required: bool
    exact_policy_digests_required_at_all_binding_points: bool
    mixed_schema_revisions_allowed: bool
    mixed_task_sets_allowed: bool
    legacy_release_path_allowed: bool
    optional_argument_release_mode_allowed: bool
    caller_asserted_success_allowed: bool
    production_profile_evidence_required: bool
    deterministic_dry_run_required: bool
    atomic_publication_required: bool
    manual_review_may_waive_automated_failure: bool


class ExportPolicy(StrictFrozenModel):
    schema_version: Literal["v7.2.export-policy.v1"]
    policy_id: NonEmptyString
    deterministic_rebuild_count: PositiveInteger
    required_join_records: tuple[
        Literal[
            "release",
            "candidate",
            "task",
            "episode",
            "trajectory",
            "sealed_output",
            "reward",
            "execution_profile",
            "split",
            "horizon",
        ],
        ...,
    ]
    excluded_statuses: tuple[
        Literal["infrastructure_error", "task_defect", "unresolved_verifier_failure"], ...
    ]
    trajectory_storage_resolution_required: bool
    trajectory_digest_verification_required: bool
    exact_output_and_reward_join_required: bool
    curriculum_from_authority_evidence_required: bool
    horizon_from_authority_evidence_required: bool
    stage_reward_event_preservation_required: bool
    termination_and_resource_metadata_preservation_required: bool
    content_addressed_shards_required: bool
    export_manifest_row_counts_required: bool
    export_manifest_tree_hashes_required: bool
    dangling_trajectory_references_allowed: bool


type Policy = (
    WorldPolicy
    | ArtifactRelationPolicy
    | WorkflowPolicy
    | CorruptionPolicy
    | HorizonPolicy
    | PublicContractPolicy
    | IndependencePolicy
    | ExploitPolicy
    | RewardPolicy
    | SourcePolicy
    | BenchmarkAnchorPolicy
    | CoveragePolicy
    | CurriculumPolicy
    | RolloutPolicy
    | BehavioralPolicy
    | ReleasePolicy
    | ExportPolicy
)


POLICY_MANIFEST_MODELS = MappingProxyType(
    {
        "world_policy.json": WorldPolicy,
        "artifact_relation_policy.json": ArtifactRelationPolicy,
        "workflow_policy.json": WorkflowPolicy,
        "corruption_policy.json": CorruptionPolicy,
        "horizon_policy.json": HorizonPolicy,
        "public_contract_policy.json": PublicContractPolicy,
        "independence_policy.json": IndependencePolicy,
        "exploit_policy.json": ExploitPolicy,
        "reward_policy.json": RewardPolicy,
        "source_policy.json": SourcePolicy,
        "benchmark_anchor_policy.json": BenchmarkAnchorPolicy,
        "coverage_policy.json": CoveragePolicy,
        "curriculum_policy.json": CurriculumPolicy,
        "rollout_policy.json": RolloutPolicy,
        "behavioral_policy.json": BehavioralPolicy,
        "release_policy.json": ReleasePolicy,
        "export_policy.json": ExportPolicy,
    }
)

POLICY_SCHEMA_MODELS = MappingProxyType(
    {
        "v7.2.world-policy.v1": WorldPolicy,
        "v7.2.artifact-relation-policy.v1": ArtifactRelationPolicy,
        "v7.2.workflow-policy.v1": WorkflowPolicy,
        "v7.2.corruption-policy.v1": CorruptionPolicy,
        "v7.2.horizon-policy.v1": HorizonPolicy,
        "v7.2.public-contract-policy.v1": PublicContractPolicy,
        "v7.2.independence-policy.v1": IndependencePolicy,
        "v7.2.exploit-policy.v1": ExploitPolicy,
        "v7.2.reward-policy.v1": RewardPolicy,
        "v7.2.source-policy.v1": SourcePolicy,
        "v7.2.benchmark-anchor-policy.v1": BenchmarkAnchorPolicy,
        "v7.2.coverage-policy.v1": CoveragePolicy,
        "v7.2.curriculum-policy.v1": CurriculumPolicy,
        "v7.2.rollout-policy.v1": RolloutPolicy,
        "v7.2.behavioral-policy.v1": BehavioralPolicy,
        "v7.2.release-policy.v1": ReleasePolicy,
        "v7.2.export-policy.v1": ExportPolicy,
    }
)


class PolicyDigestSet(StrictFrozenModel):
    schema_version: Literal["v7.2.policy-digest-set.v1"]
    world: Sha256Digest
    artifact_relation: Sha256Digest
    workflow: Sha256Digest
    corruption: Sha256Digest
    horizon: Sha256Digest
    public_contract: Sha256Digest
    independence: Sha256Digest
    exploit: Sha256Digest
    reward: Sha256Digest
    source: Sha256Digest
    benchmark_anchor: Sha256Digest
    coverage: Sha256Digest
    curriculum: Sha256Digest
    rollout: Sha256Digest
    behavioral: Sha256Digest
    release: Sha256Digest
    export: Sha256Digest


def load_policy_manifest(path: Path) -> Policy:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid policy JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"policy manifest must be a JSON object: {path}")
    schema_version = payload.get("schema_version")
    model = POLICY_SCHEMA_MODELS.get(schema_version)
    if model is None:
        raise ValueError(f"unsupported policy schema version: {schema_version!r}")
    return model.model_validate_json(raw)


def load_policy_set(root: Path) -> dict[str, Policy]:
    expected_names = set(POLICY_MANIFEST_MODELS)
    actual_names = {path.name for path in root.glob("*.json") if path.is_file()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unknown = sorted(actual_names - expected_names)
        raise ValueError(f"incomplete policy set: missing={missing}, unknown={unknown}")

    policies: dict[str, Policy] = {}
    for filename, expected_model in POLICY_MANIFEST_MODELS.items():
        policy = load_policy_manifest(root / filename)
        if not isinstance(policy, expected_model):
            raise TypeError(f"policy schema does not match manifest filename: {filename}")
        policies[filename.removesuffix("_policy.json")] = policy
    return policies


def policy_digest(policy: Policy) -> str:
    return content_digest(policy.model_dump(mode="json"))


def build_policy_digest_set(policies: dict[str, Policy]) -> PolicyDigestSet:
    expected_names = {name.removesuffix("_policy.json") for name in POLICY_MANIFEST_MODELS}
    if set(policies) != expected_names:
        missing = sorted(expected_names - set(policies))
        unknown = sorted(set(policies) - expected_names)
        raise ValueError(f"cannot bind incomplete policy set: missing={missing}, unknown={unknown}")
    return PolicyDigestSet.model_validate(
        {
            "schema_version": "v7.2.policy-digest-set.v1",
            **{name: policy_digest(policy) for name, policy in policies.items()},
        }
    )
