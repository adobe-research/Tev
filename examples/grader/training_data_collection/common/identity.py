"""Canonical serialization and content-identity primitives."""

from __future__ import annotations

import hashlib
import hmac
import json
import re

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type SeedValue = int | str | bytes

_IDENTITY_NAMESPACE = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
_MAX_NAMESPACE_LENGTH = 63
_SEED_DERIVATION_DOMAIN = b"ale-v7.2.seed.v1"


def _validate_namespace(namespace: str) -> None:
    if not isinstance(namespace, str):
        raise TypeError("identity namespace must be a string")
    if len(namespace) > _MAX_NAMESPACE_LENGTH or not _IDENTITY_NAMESPACE.fullmatch(namespace):
        raise ValueError(
            "identity namespace must be 1 to 63 lowercase alphanumeric characters "
            "separated by single dots or hyphens"
        )


def _validate_json_value(value: object, path: str = "$") -> None:
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"canonical JSON object key at {path} is not a string")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"value at {path} is not an exact JSON value")


def canonical_json(value: JsonValue) -> bytes:
    """Serialize an exact JSON value deterministically as compact UTF-8."""
    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    if type(data) is not bytes:
        raise TypeError("SHA-256 input must be bytes")
    return hashlib.sha256(data).hexdigest()


def content_digest(value: JsonValue) -> str:
    """Return the SHA-256 digest of a canonical JSON authority value."""
    return sha256_bytes(canonical_json(value))


def content_id(prefix: str, value: JsonValue) -> str:
    """Return a validated, namespaced SHA-256 content identity."""
    _validate_namespace(prefix)
    return f"{prefix}:{content_digest(value)}"


def derive_seed(root_seed: SeedValue, namespace: str) -> int:
    """Derive a deterministic 128-bit seed in a cryptographic namespace."""
    _validate_namespace(namespace)
    if type(root_seed) is int:
        if root_seed < 0:
            raise ValueError("integer root seed must be nonnegative")
        material = root_seed.to_bytes(max(1, (root_seed.bit_length() + 7) // 8), "big")
        seed_type = b"integer"
    elif type(root_seed) is str:
        if not root_seed:
            raise ValueError("string root seed must not be empty")
        material = root_seed.encode("utf-8")
        seed_type = b"string"
    elif type(root_seed) is bytes:
        if not root_seed:
            raise ValueError("bytes root seed must not be empty")
        material = root_seed
        seed_type = b"bytes"
    else:
        raise TypeError("root seed must be a nonnegative integer, nonempty string, or bytes")

    key = hashlib.sha256(
        _SEED_DERIVATION_DOMAIN + b"\x00" + seed_type + len(material).to_bytes(8, "big") + material
    ).digest()
    derived = hmac.new(
        key,
        _SEED_DERIVATION_DOMAIN + b"\x00" + namespace.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(derived[:16], "big")
