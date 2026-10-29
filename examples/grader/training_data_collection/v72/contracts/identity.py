"""Identity types and strict model base for v7.2 authority records."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from training_data_collection.common.identity import (
    JsonValue,
    canonical_json,
    content_digest,
    content_id,
    derive_seed,
    sha256_bytes,
)

IDENTITY_SCHEMA_VERSION = "v7.2.identity.v1"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
CONTENT_ID_PATTERN = r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*:[0-9a-f]{64}$"
IDENTITY_NAMESPACE_PATTERN = r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$"

Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=SHA256_PATTERN, strict=True),
]
ContentIdentifier = Annotated[
    str,
    StringConstraints(pattern=CONTENT_ID_PATTERN, strict=True),
]
IdentityNamespace = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=63,
        pattern=IDENTITY_NAMESPACE_PATTERN,
        strict=True,
    ),
]


class SeedNamespace(StrEnum):
    """Fixed independent random streams used by a v7.2 task recipe."""

    PUBLIC = "public"
    PRIVATE_WORLD = "private-world"
    HIDDEN_CASE = "hidden-case"
    PERTURBATION = "perturbation"
    BEHAVIOR = "behavior"


class StrictFrozenModel(BaseModel):
    """Base contract that rejects coercion, mutation, and unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class IdentityRecord(StrictFrozenModel):
    """A validated content identity for an authority value."""

    schema_version: Literal["v7.2.identity.v1"] = IDENTITY_SCHEMA_VERSION
    record_id: ContentIdentifier
    namespace: IdentityNamespace
    authority_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_record_id(self) -> Self:
        expected = f"{self.namespace}:{self.authority_sha256}"
        if self.record_id != expected:
            raise ValueError("record identity does not match its authority digest")
        return self

    @classmethod
    def from_authority(cls, namespace: str, authority: JsonValue) -> Self:
        """Build an identity record from the authority value itself."""
        digest = content_digest(authority)
        return cls(
            record_id=content_id(namespace, authority),
            namespace=namespace,
            authority_sha256=digest,
        )


__all__ = [
    "CONTENT_ID_PATTERN",
    "IDENTITY_NAMESPACE_PATTERN",
    "IDENTITY_SCHEMA_VERSION",
    "SHA256_PATTERN",
    "ContentIdentifier",
    "IdentityNamespace",
    "IdentityRecord",
    "SeedNamespace",
    "Sha256Digest",
    "StrictFrozenModel",
    "canonical_json",
    "content_digest",
    "content_id",
    "derive_seed",
    "sha256_bytes",
]
