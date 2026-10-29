"""Deterministic prompt rendering and governed offline authoring for v7.2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from training_data_collection.common.filesystem import iter_regular_files, tree_sha256
from training_data_collection.common.identity import canonical_json, sha256_bytes
from training_data_collection.v72.contracts.prompt import PromptGenerationAuthority
from training_data_collection.v72.contracts.public_task import (
    FrozenAuthoredPrompt,
    PublicTaskSpecification,
)

PROMPT_RENDERER_VERSION = "v72-canonical-prompt.v3"
_FIDELITY_GATE_VERSION = "v72-prompt-fidelity.v1"
_LEAKAGE_GATE_VERSION = "v72-prompt-leakage.v1"
_DECONTAMINATION_GATE_VERSION = "v72-prompt-decontamination.v1"
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_./:+-]*", re.IGNORECASE)
_MACHINE_ID = re.compile(
    r"(?:artifact-node:|(?:relation|requirement|role|rule|validator)[.:])[a-z0-9]",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NGRAM_SIZE = 8
_TEXT_CHUNK_BYTES = 1024 * 1024
_TEXT_CHUNK_OVERLAP = 1024


class CanonicalPromptError(ValueError):
    """Raised when prompt bytes do not equal the frozen public-spec rendering."""


class PromptAuthoringError(ValueError):
    """Raised when offline-authored prompt prose fails a mandatory safety gate."""


@dataclass(frozen=True)
class OfflinePromptAuthoringInput:
    """Files and fixed settings for one offline LLM prompt-authoring pass."""

    prompt_text: str
    transcript: str
    model_id: str
    benchmark_corpus_root: Path
    maximum_benchmark_similarity: float = 0.08


def _listed(values: tuple[str, ...]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def _tier_zero_prompt(specification: PublicTaskSpecification) -> str:
    environment = specification.environment
    lines = [
        f"# {specification.title}",
        "",
        (
            f"You are working on a Linux VM with the `{environment.shell}` shell. "
            f"Tools: {_listed(environment.available_tools)}. Network: disabled."
        ),
        "",
        "## Goal",
        "",
        specification.objective,
        "",
        specification.scenario,
        "",
        (
            f"## Visible Inputs (read-only, under `{environment.input_root}/`)"
            if environment.workspace_layout == "machine_contract"
            else "## Visible Inputs (read-only)"
        ),
        "",
    ]
    for public_input in specification.public_inputs:
        optional = " Optional." if not public_input.required else ""
        lines.append(
            f"- `{public_input.root}/{public_input.path}`: {public_input.description}{optional}"
        )
    lines.extend(("", "## What to do", ""))
    for step, rule in enumerate(specification.public_rules, start=1):
        lines.append(f"{step}. {rule.description}")
    lines.append("")
    for precedence in specification.authority_and_precedence_rules:
        lines.append(f"- Authority precedence: {precedence.description}")
    lines.extend(("", f"## Required Output (write under `{environment.output_root}/`)", ""))
    for output in specification.output_artifacts:
        lines.append(f"- `{environment.output_root}/{output.path}`: {output.purpose}")
    lines.append("")
    if environment.workspace_layout == "machine_contract":
        lines.append(
            "Field-level schemas and validation details for each deliverable are in "
            f"`{environment.task_specification_path}`."
        )
    else:
        lines.append(
            "Use this prompt and the staged input contracts as the complete requirements "
            "for the deliverables."
        )
    lines.append(f"Permitted dependencies: {_listed(environment.permitted_dependencies)}.")
    if specification.public_evaluation.allow_extra_files:
        lines.append(
            f"Additional supporting files may be written under `{environment.output_root}/`."
        )
    else:
        lines.append(f"Write only these deliverables under `{environment.output_root}/`.")
    immutable_instruction = f"Do not modify files under `{environment.input_root}/`."
    if environment.workspace_layout == "benchmark_native":
        immutable_instruction = (
            f"Do not modify files under `{environment.input_root}/` or "
            f"`{environment.software_root}/`."
        )
    lines.extend((immutable_instruction, ""))
    return "\n".join(lines)


def render_canonical_prompt(specification: PublicTaskSpecification) -> str:
    """Render concise contract prose or return governed authored prose verbatim."""
    if specification.authored_prompt is not None:
        require_authored_prompt_fidelity(specification, specification.authored_prompt.text)
        return specification.authored_prompt.text
    return _tier_zero_prompt(specification)


def require_canonical_prompt(
    specification: PublicTaskSpecification,
    prompt: str,
) -> None:
    """Reject any prompt that differs from the specification-derived rendering."""
    expected = render_canonical_prompt(specification)
    if prompt != expected:
        raise CanonicalPromptError("prompt.md is not the canonical rendering of public authority")


def require_authored_prompt_fidelity(
    specification: PublicTaskSpecification,
    prompt_text: str,
) -> None:
    """Require complete public obligations and reject machine-facing ID prose."""
    if _MACHINE_ID.search(prompt_text):
        raise PromptAuthoringError("authored prompt exposes internal machine identifiers")
    required_fragments = [
        f"# {specification.title}",
        "## Goal",
        specification.objective,
        specification.scenario,
        "## Visible Inputs",
        "## What to do",
        "## Required Output",
        "Network: disabled.",
        *(
            fragment
            for public_input in specification.public_inputs
            for fragment in (
                f"`{public_input.root}/{public_input.path}`",
                public_input.description,
            )
        ),
        *(rule.description for rule in specification.public_rules),
        *(rule.description for rule in specification.authority_and_precedence_rules),
        *(
            fragment
            for output in specification.output_artifacts
            for fragment in (
                f"`{specification.environment.output_root}/{output.path}`",
                output.purpose,
            )
        ),
        *(f"`{tool}`" for tool in specification.environment.available_tools),
        *(f"`{dependency}`" for dependency in specification.environment.permitted_dependencies),
    ]
    if specification.environment.workspace_layout == "machine_contract":
        required_fragments.append(f"`{specification.environment.task_specification_path}`")
        required_fragments.append(
            f"Do not modify files under `{specification.environment.input_root}/`."
        )
    else:
        required_fragments.append(
            f"Do not modify files under `{specification.environment.input_root}/` or "
            f"`{specification.environment.software_root}/`."
        )
    if not specification.public_evaluation.allow_extra_files:
        required_fragments.append(
            f"Write only these deliverables under `{specification.environment.output_root}/`."
        )
    missing = [fragment for fragment in required_fragments if fragment not in prompt_text]
    if missing:
        raise PromptAuthoringError(
            f"authored prompt omits {len(missing)} required public-contract fragments"
        )
    lowered = prompt_text.casefold()
    contradictions = (
        "network is enabled",
        "network access is allowed",
        "internet access is allowed",
    )
    if any(fragment in lowered for fragment in contradictions):
        raise PromptAuthoringError("authored prompt contradicts the disabled network policy")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).casefold().strip(".,:;")
        for match in _TOKEN.finditer(text)
        if match.group(0).strip(".,:;")
    )


def _ngrams(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {
        tokens[index : index + _NGRAM_SIZE]
        for index in range(max(0, len(tokens) - _NGRAM_SIZE + 1))
    }


def _root_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for candidate in roots:
        if candidate.is_symlink():
            raise PromptAuthoringError("prompt gate roots cannot be symlinks")
        root = candidate.resolve(strict=True)
        files.extend((root,) if root.is_file() else iter_regular_files(root))
    return tuple(files)


def _matching_tokens_and_ngrams(
    paths: tuple[Path, ...],
    candidate_tokens: set[str],
    candidate_ngrams: set[tuple[str, ...]],
) -> tuple[set[str], set[tuple[str, ...]]]:
    matching_tokens: set[str] = set()
    matching_ngrams: set[tuple[str, ...]] = set()
    for path in paths:
        overlap = ""
        with path.open("rb") as stream:
            while chunk := stream.read(_TEXT_CHUNK_BYTES):
                text = overlap + chunk.decode("utf-8", errors="ignore")
                tokens = _tokens(text)
                matching_tokens.update(candidate_tokens.intersection(tokens))
                matching_ngrams.update(candidate_ngrams.intersection(_ngrams(tokens)))
                overlap = text[-_TEXT_CHUNK_OVERLAP:]
    return matching_tokens, matching_ngrams


def require_no_private_prompt_leakage(
    specification: PublicTaskSpecification,
    prompt_text: str,
    *,
    private_roots: tuple[Path, ...],
    public_roots: tuple[Path, ...],
) -> None:
    """Reject exact private-only phrases, digests, and substantial numeric values."""
    prompt_tokens = set(_tokens(prompt_text))
    prompt_ngrams = _ngrams(_tokens(prompt_text))
    specification_text = canonical_json(
        specification.model_dump(mode="json", exclude={"authored_prompt"})
    ).decode("utf-8")
    specification_tokens = set(_tokens(specification_text))
    specification_ngrams = _ngrams(_tokens(specification_text))
    public_tokens, public_ngrams = _matching_tokens_and_ngrams(
        _root_files(public_roots),
        prompt_tokens,
        prompt_ngrams,
    )
    public_tokens.update(prompt_tokens.intersection(specification_tokens))
    public_ngrams.update(prompt_ngrams.intersection(specification_ngrams))
    private_tokens, private_ngrams = _matching_tokens_and_ngrams(
        _root_files(private_roots),
        prompt_tokens,
        prompt_ngrams,
    )
    sensitive_private_tokens = {
        token
        for token in prompt_tokens & (private_tokens - public_tokens)
        if len(token) >= 24
        or (_NUMBER.fullmatch(token) is not None and len(token.lstrip("-")) >= 4)
    }
    if sensitive_private_tokens or prompt_ngrams & (private_ngrams - public_ngrams):
        raise PromptAuthoringError("authored prompt contains private-only authority material")


def require_prompt_decontamination(
    prompt_text: str,
    benchmark_corpus_root: Path,
    *,
    maximum_similarity: float,
) -> tuple[str, float]:
    """Reject authored prose with excessive exact eight-token benchmark overlap."""
    if not 0 <= maximum_similarity <= 1:
        raise PromptAuthoringError("benchmark similarity limit must be between zero and one")
    if benchmark_corpus_root.is_symlink() or not benchmark_corpus_root.is_dir():
        raise PromptAuthoringError("benchmark corpus must be a real directory")
    corpus_root = benchmark_corpus_root.resolve(strict=True)
    corpus_paths = tuple(iter_regular_files(corpus_root))
    if not corpus_paths:
        raise PromptAuthoringError("benchmark corpus must contain regular text files")
    prompt_ngrams = _ngrams(_tokens(prompt_text))
    observed = 0.0
    for path in corpus_paths:
        _, matching_ngrams = _matching_tokens_and_ngrams((path,), set(), prompt_ngrams)
        if prompt_ngrams and matching_ngrams:
            observed = max(observed, len(matching_ngrams) / len(prompt_ngrams))
    if observed > maximum_similarity:
        raise PromptAuthoringError(
            "authored prompt exceeds the allowed benchmark-corpus similarity"
        )
    return tree_sha256(corpus_root), observed


def freeze_offline_authored_prompt(
    specification: PublicTaskSpecification,
    authoring: OfflinePromptAuthoringInput,
    *,
    private_roots: tuple[Path, ...],
    public_roots: tuple[Path, ...],
) -> PublicTaskSpecification:
    """Run all Tier 1 gates and freeze accepted LLM prose into the specification."""
    if specification.authored_prompt is not None:
        raise PromptAuthoringError("public task specification already has authored prompt prose")
    if not authoring.model_id or not authoring.transcript:
        raise PromptAuthoringError("offline LLM authoring requires a model ID and transcript")
    require_authored_prompt_fidelity(specification, authoring.prompt_text)
    require_no_private_prompt_leakage(
        specification,
        authoring.prompt_text,
        private_roots=private_roots,
        public_roots=public_roots,
    )
    benchmark_sha256, observed_similarity = require_prompt_decontamination(
        authoring.prompt_text,
        authoring.benchmark_corpus_root,
        maximum_similarity=authoring.maximum_benchmark_similarity,
    )
    frozen = FrozenAuthoredPrompt.identified(
        text=authoring.prompt_text,
        prompt_sha256=sha256_bytes(authoring.prompt_text.encode("utf-8")),
        model_id=authoring.model_id,
        temperature=0.0,
        transcript_sha256=sha256_bytes(authoring.transcript.encode("utf-8")),
        benchmark_corpus_sha256=benchmark_sha256,
        maximum_benchmark_similarity=authoring.maximum_benchmark_similarity,
        observed_benchmark_similarity=observed_similarity,
        fidelity_gate_version=_FIDELITY_GATE_VERSION,
        leakage_gate_version=_LEAKAGE_GATE_VERSION,
        decontamination_gate_version=_DECONTAMINATION_GATE_VERSION,
    )
    values = specification.model_dump(
        mode="python",
        exclude={"specification_id", "authored_prompt"},
    )
    return PublicTaskSpecification.identified(**values, authored_prompt=frozen)


def build_prompt_generation_authority(
    specification: PublicTaskSpecification,
    *,
    transcript: str | None = None,
) -> PromptGenerationAuthority:
    """Build private provenance for the exact canonical prompt bytes."""
    prompt_sha256 = sha256_bytes(render_canonical_prompt(specification).encode("utf-8"))
    if specification.authored_prompt is None:
        if transcript is not None:
            raise PromptAuthoringError("Tier 0 prompt generation cannot carry a transcript")
        return PromptGenerationAuthority.identified(
            specification_id=specification.specification_id,
            prompt_sha256=prompt_sha256,
            renderer_version=PROMPT_RENDERER_VERSION,
            generation_mode="deterministic_renderer",
        )
    authored = specification.authored_prompt
    if transcript is None:
        raise PromptAuthoringError("Tier 1 prompt generation requires its governed transcript")
    if sha256_bytes(transcript.encode("utf-8")) != authored.transcript_sha256:
        raise PromptAuthoringError("Tier 1 transcript differs from the frozen public authority")
    return PromptGenerationAuthority.identified(
        specification_id=specification.specification_id,
        prompt_sha256=prompt_sha256,
        renderer_version=PROMPT_RENDERER_VERSION,
        generation_mode="offline_llm",
        model_id=authored.model_id,
        temperature=authored.temperature,
        transcript=transcript,
        transcript_sha256=authored.transcript_sha256,
        benchmark_corpus_sha256=authored.benchmark_corpus_sha256,
    )


__all__ = [
    "PROMPT_RENDERER_VERSION",
    "CanonicalPromptError",
    "OfflinePromptAuthoringInput",
    "PromptAuthoringError",
    "build_prompt_generation_authority",
    "freeze_offline_authored_prompt",
    "render_canonical_prompt",
    "require_authored_prompt_fidelity",
    "require_canonical_prompt",
    "require_no_private_prompt_leakage",
    "require_prompt_decontamination",
]
