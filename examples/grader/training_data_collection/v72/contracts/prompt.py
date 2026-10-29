"""Private authority for deterministic and offline-authored prompt generation."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import ValidationInfo, model_validator

from training_data_collection.common.identity import content_id, sha256_bytes

from .identity import ContentIdentifier, Sha256Digest, StrictFrozenModel
from .public_task import NonEmptyString


class PromptGenerationAuthority(StrictFrozenModel):
    """Bind prompt bytes to the renderer or the governed offline transcript."""

    schema_version: Literal["v7.2.prompt-generation.v1"] = "v7.2.prompt-generation.v1"
    authority_id: ContentIdentifier
    specification_id: ContentIdentifier
    prompt_sha256: Sha256Digest
    renderer_version: Literal["v72-canonical-prompt.v3"] = "v72-canonical-prompt.v3"
    generation_mode: Literal["deterministic_renderer", "offline_llm"]
    model_id: NonEmptyString | None = None
    temperature: Literal[0.0] | None = None
    transcript: NonEmptyString | None = None
    transcript_sha256: Sha256Digest | None = None
    benchmark_corpus_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_authority(self, info: ValidationInfo) -> Self:
        authored_values = (
            self.model_id,
            self.temperature,
            self.transcript,
            self.transcript_sha256,
            self.benchmark_corpus_sha256,
        )
        if self.generation_mode == "deterministic_renderer":
            if any(value is not None for value in authored_values):
                raise ValueError("deterministic prompts cannot carry LLM authoring authority")
        elif any(value is None for value in authored_values):
            raise ValueError("offline LLM prompts require complete authoring authority")
        elif sha256_bytes((self.transcript or "").encode("utf-8")) != self.transcript_sha256:
            raise ValueError("prompt authoring transcript digest is stale")
        if not (info.context or {}).get("skip_identity") and (
            self.authority_id != self.expected_authority_id()
        ):
            raise ValueError("prompt generation authority identity is stale")
        return self

    def expected_authority_id(self) -> str:
        return content_id(
            "v72-prompt-generation",
            self.model_dump(mode="json", exclude={"authority_id"}),
        )

    @classmethod
    def identified(cls, **values: Any) -> Self:
        provisional = cls.model_validate(
            {"authority_id": f"v72-prompt-generation:{'0' * 64}", **values},
            context={"skip_identity": True},
        )
        return cls(
            authority_id=provisional.expected_authority_id(),
            **provisional.model_dump(mode="python", exclude={"authority_id"}),
        )


__all__ = ["PromptGenerationAuthority"]
