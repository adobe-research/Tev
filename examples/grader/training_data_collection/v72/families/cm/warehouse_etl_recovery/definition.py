"""World-first benchmark-shaped warehouse ETL family revision for v7.2."""

from __future__ import annotations

from pathlib import Path

from training_data_collection.common.identity import content_digest
from training_data_collection.v72.contracts.artifact import ArtifactGraph
from training_data_collection.v72.contracts.corruption import CorruptionEvent
from training_data_collection.v72.contracts.public_task import (
    AuthorityPrecedenceRule,
    JSONArtifactSchema,
    JSONFieldSpecification,
    OutputArtifact,
    PublicArtifactRelation,
    PublicEnvironment,
    PublicEvaluation,
    PublicInputRole,
    PublicRule,
    PublicTaskSpecification,
    SQLiteArtifactSchema,
    SQLiteColumnSpecification,
    SQLiteTableSpecification,
    TaskRequirement,
)
from training_data_collection.v72.contracts.workflow import (
    AllowedParallelGroup,
    VerifierEvidenceDeclaration,
    WorkflowGraph,
    WorkflowStage,
)
from training_data_collection.v72.contracts.world import LatentWorldManifest, WorldRecipe
from training_data_collection.v72.families.base import GovernedSourceView, WorldStateView
from training_data_collection.v72.generation.hidden_suite import HiddenSuiteManifest
from training_data_collection.v72.generation.world import materialize_world
from training_data_collection.v72.qualification.artifact_graph import validate_artifact_graph
from training_data_collection.v72.qualification.workflow import validate_workflow_graph

from .render import (
    CUSTOMERS_PATH,
    DATABASE_PATH,
    OUTPUT_CONTRACT_PATH,
    PRODUCTS_PATH,
    QUALITY_REPORT_PATH,
    RAW_TRANSACTIONS_PATH,
    RUNTIME_PATH,
    SCHEMA_PATH,
    SOFTWARE_PATH,
    SUMMARY_PATH,
    apply_registered_corruptions,
    build_artifact_graph,
    public_relation_id_map,
    render_consistent_artifacts,
    validate_consistent_artifacts,
)
from .world import (
    FAMILY_REVISION_ID,
    WarehouseEtlRecoveryWorldMaterializer,
    world_invariant_validators,
)

STAGE_IDS = (
    "profile-sources",
    "reconcile-facts",
    "normalize-dimensions",
    "load-facts",
    "aggregate-validate",
    "package-sidecars",
)
CRITICAL_STAGE_IDS = (
    "profile-sources",
    "reconcile-facts",
    "load-facts",
    "aggregate-validate",
    "package-sidecars",
)
PUBLIC_RULE_IDS = tuple(
    sorted(
        (
            "rule.dimensions",
            "rule.fact-load",
            "rule.ingestion",
            "rule.output-scope",
            "rule.relational-quality",
            "rule.sidecars",
        )
    )
)
REWARD_COMPONENT_IDS = tuple(
    sorted(
        (
            "reward.data-quality",
            "reward.revenue",
            "reward.row-counts",
            "reward.schema",
            "reward.sidecars",
            "reward.standardization",
        )
    )
)

_PUBLIC_ARTIFACT_IDS = {
    "customers": "input.customers",
    "output_contract": "input.output-contract",
    "products": "input.products",
    "runtime": "input.runtime",
    "schema": "input.schema",
    "transactions": "input.transactions",
    "data_quality_report": "output.data-quality-report",
    "warehouse": "output.warehouse",
    "warehouse_summary": "output.warehouse-summary",
    "software": "software.runtime-notes",
}


def _scale_counts(world: LatentWorldManifest) -> dict[str, int]:
    prefixes = {
        "customers": "scale.customers-",
        "products": "scale.products-",
        "transactions_per_month": "scale.transactions-per-month-",
    }
    counts = {
        name: int(declaration.removeprefix(prefix))
        for name, prefix in prefixes.items()
        for declaration in world.renderer_declarations
        if declaration.startswith(prefix)
    }
    if set(counts) != set(prefixes):
        raise ValueError("warehouse recovery world lacks exact scale declarations")
    return counts


def build_workflow_graph(world: LatentWorldManifest) -> WorkflowGraph:
    graph = build_artifact_graph(world)
    artifacts = {node.path: node.artifact_id for node in graph.nodes}
    relations = public_relation_id_map(graph)
    inputs = tuple(
        artifacts[path]
        for path in (
            CUSTOMERS_PATH,
            OUTPUT_CONTRACT_PATH,
            PRODUCTS_PATH,
            RAW_TRANSACTIONS_PATH,
            RUNTIME_PATH,
            SCHEMA_PATH,
            SOFTWARE_PATH,
        )
    )
    outputs = (
        artifacts[QUALITY_REPORT_PATH],
        artifacts[DATABASE_PATH],
        artifacts[SUMMARY_PATH],
    )
    counts = _scale_counts(world)
    evidence = {
        stage_id: VerifierEvidenceDeclaration.create(
            observation_kind=(
                "recomputed_artifact" if stage_id == "package-sidecars" else "state_transition"
            ),
            implementation_id=f"evidence.warehouse-recovery.{stage_id}.v1",
            observed_artifact_ids=outputs if stage_id == "package-sidecars" else (),
            observed_state_ids=(f"state.{stage_id}",),
            independently_recomputed=True,
        )
        for stage_id in STAGE_IDS
    }
    configurations = {
        "profile-sources": {
            "depends": (),
            "relations": (),
            "input_states": ("state.raw-sources",),
            "rewards": ("reward.schema",),
            "work": counts["transactions_per_month"] * 6,
        },
        "reconcile-facts": {
            "depends": ("profile-sources",),
            "relations": (relations["relation.fact-reconciliation"],),
            "input_states": ("state.profile-sources",),
            "rewards": ("reward.row-counts",),
            "work": counts["transactions_per_month"] * 6,
        },
        "normalize-dimensions": {
            "depends": ("profile-sources",),
            "relations": (relations["relation.dimension-normalization"],),
            "input_states": ("state.profile-sources",),
            "rewards": ("reward.standardization",),
            "work": counts["customers"] + counts["products"],
        },
        "load-facts": {
            "depends": ("normalize-dimensions", "reconcile-facts"),
            "relations": (relations["relation.fact-star-join"],),
            "input_states": ("state.normalize-dimensions", "state.reconcile-facts"),
            "rewards": ("reward.data-quality", "reward.revenue"),
            "work": counts["transactions_per_month"] * 6,
        },
        "aggregate-validate": {
            "depends": ("load-facts",),
            "relations": (relations["relation.relational-validation"],),
            "input_states": ("state.load-facts",),
            "rewards": ("reward.data-quality", "reward.revenue"),
            "work": counts["transactions_per_month"] * 6,
        },
        "package-sidecars": {
            "depends": ("aggregate-validate",),
            "relations": (
                relations["relation.output-scope"],
                relations["relation.sidecar-truth"],
            ),
            "input_states": ("state.aggregate-validate",),
            "rewards": ("reward.schema", "reward.sidecars"),
            "work": counts["transactions_per_month"] * 6,
        },
    }
    descriptions = {
        "profile-sources": "Profile all mixed-format source partitions and disclosed schemas.",
        "reconcile-facts": "Reconcile schema drift, timestamps, duplicates, and missing measures.",
        "normalize-dimensions": "Normalize customer and product identities and categories.",
        "load-facts": "Create surrogate keys and load formula-correct star-schema facts.",
        "aggregate-validate": "Validate native relations and recompute warehouse aggregates.",
        "package-sidecars": "Produce exact, truthful database and JSON deliverables.",
    }
    stages = tuple(
        WorkflowStage.create(
            stage_id=stage_id,
            description=descriptions[stage_id],
            depends_on_stage_ids=configurations[stage_id]["depends"],
            consumed_artifact_ids=inputs,
            consumed_relation_ids=configurations[stage_id]["relations"],
            required_tool_capabilities=("csv", "json", "sqlite"),
            input_state_ids=configurations[stage_id]["input_states"],
            produced_state_ids=(f"state.{stage_id}",),
            produced_artifact_ids=outputs if stage_id == "package-sidecars" else (),
            expected_failure_transitions=(),
            verifier_evidence_ids=(evidence[stage_id].evidence_id,),
            reward_component_ids=configurations[stage_id]["rewards"],
            minimum_semantic_work=configurations[stage_id]["work"],
            ablation_implementation_id=f"ablation.warehouse-recovery.{stage_id}.v1",
        )
        for stage_id in STAGE_IDS
    )
    workflow = WorkflowGraph.create(
        artifact_graph_id=graph.graph_id,
        curriculum_layer="family_integration",
        initial_state_ids=("state.raw-sources",),
        stages=stages,
        verifier_evidence=tuple(evidence[stage_id] for stage_id in STAGE_IDS),
        terminal_output_artifact_ids=outputs,
        critical_stage_ids=CRITICAL_STAGE_IDS,
        allowed_parallel_groups=(
            AllowedParallelGroup(stage_ids=("normalize-dimensions", "reconcile-facts")),
        ),
    )
    validate_workflow_graph(workflow, graph)
    return workflow


def _database_schema() -> SQLiteArtifactSchema:
    def columns(values: tuple[tuple[str, str], ...]) -> tuple[SQLiteColumnSpecification, ...]:
        return tuple(
            SQLiteColumnSpecification(name=name, sql_type=sql_type)  # type: ignore[arg-type]
            for name, sql_type in values
        )

    return SQLiteArtifactSchema(
        tables=(
            SQLiteTableSpecification(
                name="dim_customers",
                columns=columns(
                    (
                        ("customer_key", "INTEGER"),
                        ("customer_id", "INTEGER"),
                        ("name", "TEXT"),
                        ("email", "TEXT"),
                        ("segment", "TEXT"),
                        ("country_code", "TEXT"),
                        ("registration_date", "TEXT"),
                    )
                ),
                primary_key=("customer_key",),
            ),
            SQLiteTableSpecification(
                name="dim_products",
                columns=columns(
                    (
                        ("product_key", "INTEGER"),
                        ("product_id", "INTEGER"),
                        ("product_name", "TEXT"),
                        ("category", "TEXT"),
                        ("subcategory", "TEXT"),
                        ("base_price", "REAL"),
                        ("cost_price", "REAL"),
                        ("supplier", "TEXT"),
                        ("is_active", "INTEGER"),
                    )
                ),
                primary_key=("product_key",),
            ),
            SQLiteTableSpecification(
                name="dim_dates",
                columns=columns(
                    (
                        ("date_key", "INTEGER"),
                        ("full_date", "TEXT"),
                        ("year", "INTEGER"),
                        ("month", "INTEGER"),
                        ("day", "INTEGER"),
                        ("day_of_week", "TEXT"),
                        ("is_weekend", "INTEGER"),
                        ("quarter", "INTEGER"),
                    )
                ),
                primary_key=("date_key",),
            ),
            SQLiteTableSpecification(
                name="fact_transactions",
                columns=columns(
                    (
                        ("transaction_id", "TEXT"),
                        ("customer_key", "INTEGER"),
                        ("product_key", "INTEGER"),
                        ("date_key", "INTEGER"),
                        ("quantity", "INTEGER"),
                        ("unit_price", "REAL"),
                        ("total_amount", "REAL"),
                        ("discount_pct", "REAL"),
                        ("channel", "TEXT"),
                    )
                ),
                primary_key=("transaction_id",),
            ),
        )
    )


def _json_schema(fields: tuple[str, ...]) -> JSONArtifactSchema:
    return JSONArtifactSchema(
        fields=tuple(JSONFieldSpecification(name=name, value_type="object") for name in fields)
    )


def build_public_task_specification(
    graph: ArtifactGraph,
    workflow: WorkflowGraph,
) -> PublicTaskSpecification:
    validate_artifact_graph(graph)
    validate_workflow_graph(workflow, graph)
    inputs = (
        PublicInputRole(
            role_id="role.customers",
            artifact_id=_PUBLIC_ARTIFACT_IDS["customers"],
            path="raw_customers/customers.json",
            path_kind="file",
            artifact_type="json",
            description=(
                "Customer JSON records with country, segment, and provisional-record drift."
            ),
            critical=True,
            workflow_stage_ids=("normalize-dimensions", "package-sidecars"),
        ),
        PublicInputRole(
            role_id="role.output-contract",
            artifact_id=_PUBLIC_ARTIFACT_IDS["output_contract"],
            path="output_contract.json",
            path_kind="file",
            artifact_type="json",
            description="Exact required filenames and JSON sidecar section and field contract.",
            critical=True,
            workflow_stage_ids=("package-sidecars",),
        ),
        PublicInputRole(
            role_id="role.products",
            artifact_id=_PUBLIC_ARTIFACT_IDS["products"],
            path="raw_products/product_catalog.tsv",
            path_kind="file",
            artifact_type="tsv",
            description="Product TSV records with supplier, boolean, and category drift.",
            critical=True,
            workflow_stage_ids=("normalize-dimensions", "package-sidecars"),
        ),
        PublicInputRole(
            role_id="role.runtime",
            artifact_id=_PUBLIC_ARTIFACT_IDS["runtime"],
            path="runtime_env",
            path_kind="directory",
            artifact_type="toml",
            description="Offline Python runtime manifest.",
            workflow_stage_ids=("package-sidecars",),
        ),
        PublicInputRole(
            role_id="role.schema",
            artifact_id=_PUBLIC_ARTIFACT_IDS["schema"],
            path="target_schema_spec.json",
            path_kind="file",
            artifact_type="json",
            description=(
                "Authoritative star-schema tables, columns, value domains, date range, and "
                "customer preservation rules."
            ),
            critical=True,
            workflow_stage_ids=(
                "aggregate-validate",
                "normalize-dimensions",
                "reconcile-facts",
            ),
        ),
        PublicInputRole(
            role_id="role.transactions",
            artifact_id=_PUBLIC_ARTIFACT_IDS["transactions"],
            path="raw_transactions",
            path_kind="directory",
            artifact_type="csv",
            description=(
                "Six monthly transaction CSV partitions with schema drift, duplicates, missing "
                "measures, missing customer IDs, and several timestamp formats."
            ),
            critical=True,
            workflow_stage_ids=(
                "aggregate-validate",
                "load-facts",
                "reconcile-facts",
            ),
        ),
        PublicInputRole(
            role_id="role.software",
            artifact_id=_PUBLIC_ARTIFACT_IDS["software"],
            root="software",
            path="README.txt",
            path_kind="file",
            artifact_type="text",
            description="Local Python and SQLite usage notes.",
            workflow_stage_ids=("package-sidecars",),
        ),
    )
    outputs = (
        OutputArtifact(
            artifact_id=_PUBLIC_ARTIFACT_IDS["data_quality_report"],
            path="data_quality_report.json",
            purpose=(
                "Truthful extraction, transformation, load, and quality-check evidence matching "
                "the source files and final database."
            ),
            native_schema=_json_schema(("extraction", "transformations", "load", "quality_checks")),
            public_rule_ids=("rule.output-scope", "rule.sidecars"),
            produced_by_stage_ids=("package-sidecars",),
        ),
        OutputArtifact(
            artifact_id=_PUBLIC_ARTIFACT_IDS["warehouse"],
            path="warehouse.db",
            purpose="SQLite star-schema warehouse with three dimensions and one fact table.",
            native_schema=_database_schema(),
            public_rule_ids=tuple(
                sorted(
                    (
                        "rule.dimensions",
                        "rule.fact-load",
                        "rule.ingestion",
                        "rule.output-scope",
                        "rule.relational-quality",
                    )
                )
            ),
            produced_by_stage_ids=("package-sidecars",),
        ),
        OutputArtifact(
            artifact_id=_PUBLIC_ARTIFACT_IDS["warehouse_summary"],
            path="warehouse_summary.json",
            purpose="Truthful fact and dimension aggregates recomputed from warehouse.db.",
            native_schema=_json_schema(("fact_table", "dim_customers", "dim_products")),
            public_rule_ids=("rule.output-scope", "rule.sidecars"),
            produced_by_stage_ids=("package-sidecars",),
        ),
    )
    relation_definitions = (
        (
            "relation.dimension-normalization",
            "derived_from",
            ("customers", "products", "schema"),
            ("warehouse",),
            "Normalize customer and product dimensions from their authoritative value domains.",
            "normalize-dimensions",
            "rule.dimensions",
        ),
        (
            "relation.fact-reconciliation",
            "generated_from",
            ("transactions", "schema"),
            ("warehouse",),
            "Reconcile every partition before loading accepted transaction identities.",
            "reconcile-facts",
            "rule.ingestion",
        ),
        (
            "relation.fact-star-join",
            "joins_on",
            ("transactions", "customers", "products"),
            ("warehouse",),
            "Resolve every accepted fact to customer, product, and date surrogate keys.",
            "load-facts",
            "rule.fact-load",
        ),
        (
            "relation.output-scope",
            "constrained_by",
            ("output_contract", "runtime", "software"),
            ("warehouse", "data_quality_report", "warehouse_summary"),
            "The output and runtime contracts determine the exact deliverable boundary.",
            "package-sidecars",
            "rule.output-scope",
        ),
        (
            "relation.relational-validation",
            "validated_by",
            ("schema", "transactions"),
            ("warehouse",),
            "Validate uniqueness, formulas, date coverage, and foreign-key resolution.",
            "aggregate-validate",
            "rule.relational-quality",
        ),
        (
            "relation.sidecar-truth",
            "summarizes",
            ("transactions", "customers", "products", "output_contract"),
            ("data_quality_report", "warehouse_summary"),
            "Both JSON sidecars must truthfully summarize source defects and database contents.",
            "package-sidecars",
            "rule.sidecars",
        ),
    )
    relations = tuple(
        PublicArtifactRelation(
            relation_id=relation_id,
            relation_type=relation_type,  # type: ignore[arg-type]
            source_artifact_ids=tuple(_PUBLIC_ARTIFACT_IDS[name] for name in source_names),
            target_artifact_ids=tuple(_PUBLIC_ARTIFACT_IDS[name] for name in target_names),
            description=description,
            workflow_stage_ids=(stage_id,),
            public_rule_ids=(rule_id,),
            validator_id=f"validator.{relation_id.removeprefix('relation.')}",
        )
        for (
            relation_id,
            relation_type,
            source_names,
            target_names,
            description,
            stage_id,
            rule_id,
        ) in relation_definitions
    )
    rules = (
        PublicRule(
            rule_id="rule.dimensions",
            description=(
                "Normalize segments to basic, premium, standard, or unknown; countries to CA, "
                "DE, GB, US, or XX; suppliers to the published canonical names; activity flags "
                "to 0 or 1; and blank product categories to Uncategorized. Preserve every listed "
                "provisional customer with documented fallbacks and add exactly one Unknown "
                "Customer row with customer_id 0."
            ),
            validator_id="validator.dimensions",
            input_role_ids=("role.customers", "role.products", "role.schema"),
            target_artifact_ids=(_PUBLIC_ARTIFACT_IDS["warehouse"],),
            relation_ids=("relation.dimension-normalization",),
            workflow_stage_ids=("normalize-dimensions",),
        ),
        PublicRule(
            rule_id="rule.ingestion",
            description=(
                "Read all six monthly CSV files, fill missing discount_pct with 0 and channel "
                "with unknown, normalize all published timestamp formats, keep the first row for "
                "each duplicate transaction_id, drop rows missing quantity or unit_price, and map "
                "blank customer IDs to customer_id 0."
            ),
            validator_id="validator.ingestion",
            input_role_ids=("role.schema", "role.transactions"),
            target_artifact_ids=(_PUBLIC_ARTIFACT_IDS["warehouse"],),
            relation_ids=("relation.fact-reconciliation",),
            workflow_stage_ids=("reconcile-facts",),
        ),
        PublicRule(
            rule_id="rule.fact-load",
            description=(
                "Create deterministic customer and product surrogate keys, derive YYYYMMDD date "
                "keys, resolve every accepted fact to all dimensions, and recompute total_amount "
                "as quantity times unit_price times one minus discount_pct rounded to cents."
            ),
            validator_id="validator.fact-load",
            input_role_ids=("role.customers", "role.products", "role.transactions"),
            target_artifact_ids=(_PUBLIC_ARTIFACT_IDS["warehouse"],),
            relation_ids=("relation.fact-star-join",),
            workflow_stage_ids=("load-facts",),
        ),
        PublicRule(
            rule_id="rule.output-scope",
            description=(
                "Write exactly warehouse.db, data_quality_report.json, and "
                "warehouse_summary.json directly under output. Do not modify input or software."
            ),
            validator_id="validator.output-scope",
            input_role_ids=("role.output-contract", "role.runtime", "role.software"),
            target_artifact_ids=tuple(item.artifact_id for item in outputs),
            relation_ids=("relation.output-scope",),
            workflow_stage_ids=("package-sidecars",),
        ),
        PublicRule(
            rule_id="rule.relational-quality",
            description=(
                "Build every date from 2025-01-01 through 2025-06-30 inclusive, enforce unique "
                "transaction identities, keep required measures non-null and finite, preserve "
                "formula correctness, and leave no unresolved foreign keys."
            ),
            validator_id="validator.relational-quality",
            input_role_ids=("role.schema", "role.transactions"),
            target_artifact_ids=(_PUBLIC_ARTIFACT_IDS["warehouse"],),
            relation_ids=("relation.relational-validation",),
            workflow_stage_ids=("aggregate-validate",),
        ),
        PublicRule(
            rule_id="rule.sidecars",
            description=(
                "Populate every field listed in output_contract.json. Recompute extraction and "
                "defect counts from the staged sources and all load counts, quality flags, "
                "distributions, date bounds, averages, and revenue from the final database so "
                "both sidecars remain complete and truthful."
            ),
            validator_id="validator.sidecars",
            input_role_ids=(
                "role.customers",
                "role.output-contract",
                "role.products",
                "role.transactions",
            ),
            target_artifact_ids=(
                _PUBLIC_ARTIFACT_IDS["data_quality_report"],
                _PUBLIC_ARTIFACT_IDS["warehouse_summary"],
            ),
            relation_ids=("relation.sidecar-truth",),
            workflow_stage_ids=("package-sidecars",),
        ),
    )
    role_by_artifact = {item.artifact_id: item.role_id for item in inputs}
    rule_by_id = {item.rule_id: item for item in rules}
    requirements = []
    for step, relation in enumerate(relations, start=1):
        rule = rule_by_id[relation.public_rule_ids[0]]
        requirements.append(
            TaskRequirement(
                requirement_id=f"requirement.{rule.rule_id.removeprefix('rule.')}",
                ordered_step=step,
                description=rule.description,
                input_role_ids=tuple(
                    role_by_artifact[artifact_id]
                    for artifact_id in relation.source_artifact_ids
                    if artifact_id in role_by_artifact
                ),
                relation_ids=(relation.relation_id,),
                workflow_stage_ids=relation.workflow_stage_ids,
                output_artifact_ids=relation.target_artifact_ids,
                public_rule_ids=(rule.rule_id,),
            )
        )
    return PublicTaskSpecification.identified(
        title="Retail Warehouse ETL Recovery",
        objective=(
            "Build a clean star-schema SQLite warehouse from six months of messy retail data "
            "and produce two truthful JSON sidecars."
        ),
        scenario=(
            "The staged transaction partitions, customer records, and product catalog contain "
            "schema drift, duplicates, missing values, and categorical variants. Reconcile them "
            "using the supplied schema and output contracts."
        ),
        public_inputs=inputs,
        ordered_requirements=tuple(requirements),
        output_artifacts=outputs,
        public_rules=rules,
        public_artifact_relations=relations,
        authority_and_precedence_rules=(
            AuthorityPrecedenceRule(
                precedence_id="precedence.schema",
                description=(
                    "target_schema_spec.json overrides raw spelling and representation variants."
                ),
                higher_authority_artifact_id=_PUBLIC_ARTIFACT_IDS["schema"],
                lower_authority_artifact_ids=(
                    _PUBLIC_ARTIFACT_IDS["customers"],
                    _PUBLIC_ARTIFACT_IDS["products"],
                    _PUBLIC_ARTIFACT_IDS["transactions"],
                ),
                public_rule_ids=("rule.dimensions",),
                workflow_stage_ids=("normalize-dimensions",),
                validator_id="validator.schema-precedence",
            ),
        ),
        environment=PublicEnvironment(
            workspace_layout="benchmark_native",
            task_specification_path="input/task_prompt.md",
            available_tools=("python3", "sqlite3"),
            permitted_dependencies=("python", "sqlite"),
        ),
        public_evaluation=PublicEvaluation(
            allow_extra_files=False,
            validation_command=(
                "python3",
                "-m",
                (
                    "training_data_collection.v72.families.cm."
                    "warehouse_etl_recovery.public_validation"
                ),
            ),
            public_rule_ids=PUBLIC_RULE_IDS,
            default_numeric_tolerance=0.01,
        ),
    )


class WarehouseEtlRecoveryFamily:
    implementation_digest = content_digest(
        {"family": FAMILY_REVISION_ID, "implementation": "world-first-dag", "version": 1}
    )

    def build_world(
        self,
        recipe: WorldRecipe,
        private_root: Path,
        governed_sources: GovernedSourceView,
    ) -> LatentWorldManifest:
        sources = {
            source_id: governed_sources.read_source_bytes(source_id)
            for source_id in recipe.source_record_ids
        }
        return materialize_world(
            recipe=recipe,
            materializer=WarehouseEtlRecoveryWorldMaterializer(),
            destination=private_root / "world",
            invariant_validators=world_invariant_validators(),
            governed_sources=sources,
        ).manifest

    def build_workflow(self, world: LatentWorldManifest) -> WorkflowGraph:
        return build_workflow_graph(world)

    def render_artifacts(
        self,
        world: LatentWorldManifest,
        workflow: WorkflowGraph,
        public_root: Path,
        private_root: Path,
    ) -> ArtifactGraph:
        view = WorldStateView(private_root, world)
        render_consistent_artifacts(view, public_root)
        if not validate_consistent_artifacts(view, public_root):
            raise ValueError("warehouse baseline projections differ from their latent world")
        graph = build_artifact_graph(world)
        if workflow.artifact_graph_id != graph.graph_id:
            raise ValueError("warehouse workflow differs from its rendered artifact graph")
        return graph

    def apply_corruptions(
        self,
        world: LatentWorldManifest,
        graph: ArtifactGraph,
        public_root: Path,
    ) -> tuple[ArtifactGraph, tuple[CorruptionEvent, ...]]:
        return apply_registered_corruptions(world, graph, public_root)

    def build_public_specification(
        self,
        graph: ArtifactGraph,
        workflow: WorkflowGraph,
    ) -> PublicTaskSpecification:
        return build_public_task_specification(graph, workflow)

    def build_reward_contract(self):
        from .reward import build_reward_contract

        return build_reward_contract()

    def build_progress_contract(self, graph: ArtifactGraph, workflow: WorkflowGraph):
        from .progress import build_progress_contract

        return build_progress_contract(graph, workflow)

    def build_hidden_suite(
        self,
        recipe: WorldRecipe,
        world: LatentWorldManifest,
        private_root: Path,
    ) -> HiddenSuiteManifest:
        from .hidden_suite import materialize_hidden_suite

        graph = build_artifact_graph(world)
        return materialize_hidden_suite(
            world=world,
            private_root=private_root,
            graph=graph,
            hidden_case_seed=recipe.hidden_case_seed,
            relation_ids=tuple(sorted(relation.relation_id for relation in graph.relations)),
            destination=private_root / "hidden_suite",
        )


WAREHOUSE_ETL_RECOVERY_FAMILY = WarehouseEtlRecoveryFamily()


__all__ = [
    "CRITICAL_STAGE_IDS",
    "PUBLIC_RULE_IDS",
    "REWARD_COMPONENT_IDS",
    "STAGE_IDS",
    "WAREHOUSE_ETL_RECOVERY_FAMILY",
    "WarehouseEtlRecoveryFamily",
    "build_public_task_specification",
    "build_workflow_graph",
]
