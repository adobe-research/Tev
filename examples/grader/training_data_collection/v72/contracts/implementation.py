"""Content-bound implementation authority contracts for v7.2 qualification."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from training_data_collection.common.identity import canonical_json, content_id
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)

type ImplementationAuthorityKind = Literal[
    "materializer",
    "reference",
    "alternative",
    "verifier",
]

_MODULE_ID = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_ENTRYPOINT = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):"
    r"(?P<qualname>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)
NonEmptyString = Annotated[str, Field(min_length=1)]


class _ImplementationAuthorityValue(StrictFrozenModel):
    schema_version: Literal["v7.2.implementation-authority.v1"] = "v7.2.implementation-authority.v1"
    authority_kind: ImplementationAuthorityKind
    source_tree_sha256: Sha256Digest
    entrypoint: NonEmptyString
    dependency_closure_sha256: Sha256Digest
    allowed_shared_module_ids: tuple[NonEmptyString, ...]
    algorithm_description_digest: Sha256Digest

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        if _ENTRYPOINT.fullmatch(value) is None:
            raise ValueError("entrypoint must be an importable module:qualified.name")
        return value

    @field_validator("allowed_shared_module_ids")
    @classmethod
    def validate_shared_modules(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("allowed shared module IDs must be sorted and unique")
        if any(_MODULE_ID.fullmatch(module_id) is None for module_id in value):
            raise ValueError("allowed shared module IDs must be absolute Python module names")
        return value


class ImplementationAuthority(_ImplementationAuthorityValue):
    """Immutable source, dependency, entrypoint, and algorithm authority."""

    authority_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.authority_id != self.expected_authority_id():
            raise ValueError("authority_id does not match implementation authority")
        return self

    def expected_authority_id(self) -> str:
        """Recompute the content identity from every authority-bearing field."""
        return content_id(
            "v72-implementation",
            self.model_dump(mode="json", exclude={"authority_id"}),
        )

    @classmethod
    def identified(
        cls,
        *,
        authority_kind: ImplementationAuthorityKind,
        source_tree_sha256: str,
        entrypoint: str,
        dependency_closure_sha256: str,
        allowed_shared_module_ids: tuple[str, ...],
        algorithm_description_digest: str,
    ) -> Self:
        """Validate the authority value and derive its content identity."""
        value = _ImplementationAuthorityValue(
            authority_kind=authority_kind,
            source_tree_sha256=source_tree_sha256,
            entrypoint=entrypoint,
            dependency_closure_sha256=dependency_closure_sha256,
            allowed_shared_module_ids=allowed_shared_module_ids,
            algorithm_description_digest=algorithm_description_digest,
        )
        payload = value.model_dump(mode="json")
        return cls.model_validate_json(
            canonical_json({"authority_id": content_id("v72-implementation", payload), **payload})
        )


__all__ = ["ImplementationAuthority", "ImplementationAuthorityKind"]
