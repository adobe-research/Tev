"""Strict source-lineage and split authority for v7.2 packages."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from training_data_collection.common.identity import (
    canonical_json,
    content_digest,
    content_id,
    sha256_bytes,
)

from .identity import ContentIdentifier, Sha256Digest, StrictFrozenModel

GENERATED_SOURCE_PAYLOAD_PATH = "generator_authority.json"


def generated_source_payload_bytes(*, origin_identity: str, immutable_version: str) -> bytes:
    """Return deterministic generated-source producer bytes without split authority."""
    return canonical_json(
        {
            "schema_version": "v7.2.generated-source-payload.v1",
            "origin_identity": origin_identity,
            "immutable_version": immutable_version,
        }
    )


def generated_source_tree_sha256(*, origin_identity: str, immutable_version: str) -> str:
    """Return the exact one-file tree digest for deterministic generator authority."""
    payload = generated_source_payload_bytes(
        origin_identity=origin_identity,
        immutable_version=immutable_version,
    )
    return content_digest(
        [
            {
                "path": GENERATED_SOURCE_PAYLOAD_PATH,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        ]
    )


class _SourceRecordAuthority(StrictFrozenModel):
    schema_version: Literal["v7.2.source-record.v1"] = "v7.2.source-record.v1"
    source_kind: Literal[
        "generated",
        "repository",
        "dataset",
        "paper",
        "packet_capture",
        "compiled_binary",
        "checkpoint",
        "live_system",
    ]
    origin_identity: str = Field(min_length=1)
    immutable_version: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    license_evidence_id: ContentIdentifier
    acquisition_record_id: ContentIdentifier
    transformation_record_ids: tuple[ContentIdentifier, ...] = Field(min_length=1)
    source_tree_sha256: Sha256Digest
    split: Literal["train", "validation", "held_out"]
    split_group: str = Field(min_length=1)
    redistribution: Literal["permitted", "restricted", "prohibited"]

    @model_validator(mode="after")
    def validate_source_authority(self) -> Self:
        for label, value in (
            ("origin identity", self.origin_identity),
            ("immutable version", self.immutable_version),
            ("license identity", self.license_id),
            ("split group", self.split_group),
        ):
            if not value.strip():
                raise ValueError(f"source {label} cannot be blank")
        if self.transformation_record_ids != tuple(sorted(self.transformation_record_ids)):
            raise ValueError("source transformation records must be sorted")
        if len(self.transformation_record_ids) != len(set(self.transformation_record_ids)):
            raise ValueError("source transformation records must be unique")
        if self.source_tree_sha256 == "0" * 64:
            raise ValueError("source tree digest cannot use the sentinel digest")
        if self.source_kind == "generated" and not self.origin_identity.startswith("generated:"):
            raise ValueError("generated source identities must use the generated namespace")
        return self


class SourceRecord(_SourceRecordAuthority):
    """Content-bound origin, license, transformation, and split record."""

    source_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_source_id(self) -> Self:
        if self.source_id != self.expected_source_id():
            raise ValueError("source_id does not match source authority")
        return self

    def expected_source_id(self) -> str:
        return content_id(
            "v72-source",
            self.model_dump(mode="json", exclude={"source_id"}),
        )

    @classmethod
    def identified(cls, **values: object) -> SourceRecord:
        authority = _SourceRecordAuthority.model_validate(values)
        payload = authority.model_dump(mode="python")
        return cls(
            **payload,
            source_id=content_id("v72-source", authority.model_dump(mode="json")),
        )


class SourcePayloadAuthority(StrictFrozenModel):
    """Exact realized bytes and producer authority for one governed source record."""

    schema_version: Literal["v7.2.source-payload-authority.v1"] = "v7.2.source-payload-authority.v1"
    payload_authority_id: ContentIdentifier
    source_id: ContentIdentifier
    source_tree_sha256: Sha256Digest
    acquisition_record_id: ContentIdentifier
    realization_kind: Literal["deterministic_generator", "immutable_snapshot"]
    realization_authority_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.payload_authority_id != self.expected_payload_authority_id():
            raise ValueError("source payload authority identity is stale")
        return self

    def expected_payload_authority_id(self) -> str:
        return content_id(
            "v72-source-payload-authority",
            self.model_dump(mode="json", exclude={"payload_authority_id"}),
        )

    @classmethod
    def identified(
        cls,
        *,
        source_record: SourceRecord,
        realization_kind: str,
        realization_authority_id: str,
    ) -> SourcePayloadAuthority:
        if (source_record.source_kind == "generated") != (
            realization_kind == "deterministic_generator"
        ):
            raise ValueError(
                "generated sources require a deterministic generator and other sources "
                "require an immutable snapshot"
            )
        values = {
            "schema_version": "v7.2.source-payload-authority.v1",
            "source_id": source_record.source_id,
            "source_tree_sha256": source_record.source_tree_sha256,
            "acquisition_record_id": source_record.acquisition_record_id,
            "realization_kind": realization_kind,
            "realization_authority_id": realization_authority_id,
        }
        return cls(
            payload_authority_id=content_id("v72-source-payload-authority", values),
            **values,
        )


__all__ = [
    "GENERATED_SOURCE_PAYLOAD_PATH",
    "SourcePayloadAuthority",
    "SourceRecord",
    "generated_source_payload_bytes",
    "generated_source_tree_sha256",
]
