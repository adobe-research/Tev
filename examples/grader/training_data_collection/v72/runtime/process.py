"""Fresh-process JSON execution boundary with content-bound evidence."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

import psutil
from pydantic import Field, field_validator, model_validator

from training_data_collection.common.filesystem import file_sha256, tree_sha256
from training_data_collection.common.identity import (
    JsonValue,
    canonical_json,
    content_digest,
    content_id,
    sha256_bytes,
)
from training_data_collection.v72.contracts.identity import (
    ContentIdentifier,
    Sha256Digest,
    StrictFrozenModel,
)

type JsonProcessStatus = Literal[
    "success",
    "policy_failure",
    "resource_failure",
    "timeout",
    "infrastructure_failure",
]
type JsonProcessIsolation = Literal["none", "policy", "verifier"]

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ABSOLUTE_PATH_FRAGMENT = re.compile(
    r"(^|[\s=\"'(:])/(?!/)(?:[^\s\"'<>,;]+)",
)
_ENTRYPOINT = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):"
    r"(?P<qualname>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)
_CHILD_POLICY_FAILURE = 20
_CHILD_RESOURCE_FAILURE = 75
_CHILD_INFRASTRUCTURE_FAILURE = 70
_UNSHARE = Path("/usr/bin/unshare")
_PRLIMIT = Path("/usr/bin/prlimit")
_SANDBOX_PROGRAM = r"""
set -eu
sandbox_root=$1
working_directory=$2
read_count=$3
shift 3
mount --make-rprivate /
mount -t tmpfs -o mode=0755 tmpfs "$sandbox_root"
mkdir -p "$sandbox_root/tmp"
mount -t tmpfs -o mode=1777 tmpfs "$sandbox_root/tmp"
i=0
while [ "$i" -lt "$read_count" ]; do
    source_path=$1
    shift
    target_path="$sandbox_root$source_path"
    if [ -d "$source_path" ]; then
        mkdir -p "$target_path"
    else
        mkdir -p "$(dirname "$target_path")"
        : > "$target_path"
    fi
    if [ -d "$source_path" ]; then
        mount --rbind "$source_path" "$target_path"
        mount --make-rprivate "$target_path"
        findmnt -Rln -o TARGET --target "$target_path" | tac | while IFS= read -r nested_target; do
            mount -o remount,bind,ro "$nested_target"
        done
    else
        mount --bind "$source_path" "$target_path"
        mount -o remount,bind,ro "$target_path"
    fi
    i=$((i + 1))
done
write_count=$1
shift
i=0
while [ "$i" -lt "$write_count" ]; do
    source_path=$1
    shift
    target_path="$sandbox_root$source_path"
    mkdir -p "$target_path"
    mount --rbind "$source_path" "$target_path"
    i=$((i + 1))
done
runtime_alias=$1
runtime_target=$2
shift 2
if [ "$runtime_alias" != "$runtime_target" ]; then
    mkdir -p "$(dirname "$sandbox_root$runtime_alias")"
    ln -s "$runtime_target" "$sandbox_root$runtime_alias"
fi
shift
if [ ! -e "$sandbox_root/lib" ] && [ -d "$sandbox_root/usr/lib" ]; then
    ln -s usr/lib "$sandbox_root/lib"
fi
if [ ! -e "$sandbox_root/lib64" ] && [ -d "$sandbox_root/usr/lib64" ]; then
    ln -s usr/lib64 "$sandbox_root/lib64"
fi
working_parent="$(dirname "$sandbox_root$working_directory")"
if ! chmod 0555 "$working_parent" 2>/dev/null; then
    findmnt -n -o OPTIONS --target "$working_parent" | tr ',' '\n' | grep -qx ro
fi
exec chroot "$sandbox_root" /usr/bin/env -C "$working_directory" "$@"
"""
_CHILD_PROGRAM = r"""
import contextlib
import ctypes
import importlib
import json
import sys

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(38, 1, 0, 0, 0) != 0:
    raise SystemExit(70)
for capability in range(64):
    libc.prctl(24, capability, 0, 0, 0)

class CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

class CapabilityData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]

header = CapabilityHeader(0x20080522, 0)
data = (CapabilityData * 2)()
if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
    raise SystemExit(70)

entrypoint = sys.argv[1]
module_name, qualname = entrypoint.split(":", 1)
try:
    with contextlib.redirect_stdout(sys.stderr):
        target = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    if not callable(target):
        raise TypeError("entrypoint does not resolve to a callable")
    payload = json.loads(sys.stdin.buffer.read())
except Exception as error:
    sys.stderr.write(json.dumps({"phase": "boundary", "type": type(error).__name__, "detail": str(error)}, sort_keys=True))
    raise SystemExit(70)

try:
    with contextlib.redirect_stdout(sys.stderr):
        response = target(payload)
except MemoryError as error:
    sys.stderr.write(json.dumps({"phase": "call", "type": type(error).__name__, "detail": str(error)}, sort_keys=True))
    raise SystemExit(75)
except Exception as error:
    sys.stderr.write(json.dumps({"phase": "call", "type": type(error).__name__, "detail": str(error)}, sort_keys=True))
    raise SystemExit(20)

try:
    encoded = json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
except Exception as error:
    sys.stderr.write(json.dumps({"phase": "response", "type": type(error).__name__, "detail": str(error)}, sort_keys=True))
    raise SystemExit(70)
sys.stdout.buffer.write(encoded)
"""

NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveTimeout = Annotated[float, Field(gt=0.0, le=3_600.0)]
PositiveBytes = Annotated[int, Field(gt=0)]


def _mount_authority_sha256(path: Path) -> str:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if stat.S_ISCHR(metadata.st_mode):
        return content_digest(
            {
                "source_kind": "device",
                "device_rdev": metadata.st_rdev,
            }
        )
    runtime_roots = {
        candidate.resolve(strict=True)
        for candidate in (
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            Path(sys.prefix),
            Path(sys.executable).resolve(strict=True).parents[1],
        )
        if candidate.exists()
    }
    if resolved in runtime_roots:
        return content_digest(
            {
                "source_kind": "runtime_directory",
                "device_id": metadata.st_dev,
                "inode": metadata.st_ino,
            }
        )
    if resolved.is_file():
        return file_sha256(resolved)
    if resolved.is_dir():
        return tree_sha256(resolved)
    raise ValueError("sandbox mount authority is not a file, directory, or device")


class ProcessEnvironmentVariable(StrictFrozenModel):
    """One explicit environment binding supplied to a child process."""

    name: NonEmptyString
    value: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if _ENVIRONMENT_NAME.fullmatch(value) is None:
            raise ValueError("environment variable name is invalid")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("environment variable values cannot contain NUL")
        return value


class ProcessMountAuthority(StrictFrozenModel):
    """Host-path-free content or device authority for one sandbox mount."""

    logical_role: NonEmptyString
    access_mode: Literal["read_only", "read_write"]
    source_kind: Literal["file", "tree", "device", "runtime_directory"]
    authority_sha256: Sha256Digest
    device_rdev: int | None = None

    @model_validator(mode="after")
    def validate_device(self) -> Self:
        if (self.source_kind == "device") != (self.device_rdev is not None):
            raise ValueError("sandbox device mount authority requires exact device identity")
        return self


class _JsonProcessRequestValue(StrictFrozenModel):
    schema_version: Literal["v7.2.json-process-request.v1"] = "v7.2.json-process-request.v1"
    entrypoint: NonEmptyString
    payload: JsonValue
    authority_context: JsonValue | None = None
    cwd: NonEmptyString
    environment: tuple[ProcessEnvironmentVariable, ...]
    timeout_seconds: PositiveTimeout
    isolation: JsonProcessIsolation = "none"
    read_only_roots: tuple[NonEmptyString, ...] = ()
    writable_roots: tuple[NonEmptyString, ...] = ()
    network_access: Literal[False] = False
    maximum_cpu_seconds: PositiveTimeout = 3_600.0
    maximum_memory_bytes: PositiveBytes = 4 * 2**30
    maximum_file_bytes: PositiveBytes = 4 * 2**30
    maximum_output_entries: PositiveBytes = 10_000
    maximum_capture_bytes: PositiveBytes = 2**20
    code_closure_sha256: Sha256Digest | None = None
    runtime_device_rdevs: tuple[int, ...] = ()
    read_root_authority_sha256s: tuple[Sha256Digest, ...] = ()
    writable_root_authority_sha256s: tuple[Sha256Digest, ...] = ()

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        if _ENTRYPOINT.fullmatch(value) is None:
            raise ValueError("entrypoint must be an importable module:qualified.name")
        return value

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("process cwd must be absolute")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("process cwd cannot contain control characters")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls,
        value: tuple[ProcessEnvironmentVariable, ...],
    ) -> tuple[ProcessEnvironmentVariable, ...]:
        names = tuple(item.name for item in value)
        if names != tuple(sorted(set(names))):
            raise ValueError("process environment must be sorted with unique names")
        return value

    @model_validator(mode="after")
    def validate_isolation(self) -> Self:
        for roots, label in (
            (self.read_only_roots, "read-only"),
            (self.writable_roots, "writable"),
        ):
            if roots != tuple(sorted(set(roots))):
                raise ValueError(f"sandbox {label} roots must be sorted and unique")
            if any(not Path(root).is_absolute() for root in roots):
                raise ValueError(f"sandbox {label} roots must be absolute")
            if any(
                any(ord(character) < 32 or ord(character) == 127 for character in root)
                for root in roots
            ):
                raise ValueError(f"sandbox {label} roots cannot contain control characters")
        if self.isolation == "none" and (self.read_only_roots or self.writable_roots):
            raise ValueError("non-isolated execution cannot declare sandbox mounts")
        if self.isolation != "none" and not self.read_only_roots:
            raise ValueError("isolated execution requires explicit read roots")
        if (self.isolation == "none") != (self.code_closure_sha256 is None):
            raise ValueError("isolated execution requires exact staged code authority")
        if self.isolation == "policy" and not self.writable_roots:
            raise ValueError("isolated policy execution requires an explicit write root")
        for writable in self.writable_roots:
            if not any(Path(writable).is_relative_to(Path(root)) for root in self.read_only_roots):
                raise ValueError("sandbox writable roots must be within an allowed read root")
        device_roots = tuple(
            root
            for root in self.read_only_roots
            if Path(root) in {Path("/dev/null"), Path("/dev/random"), Path("/dev/urandom")}
        )
        if len(device_roots) != len(self.runtime_device_rdevs):
            raise ValueError("runtime device identities do not match declared device roots")
        if len(self.read_only_roots) != len(self.read_root_authority_sha256s):
            raise ValueError("sandbox read roots lack exact mount authority digests")
        if len(self.writable_roots) != len(self.writable_root_authority_sha256s):
            raise ValueError("sandbox writable roots lack exact mount authority digests")
        return self


class JsonProcessRequest(_JsonProcessRequestValue):
    """Immutable request for invoking one JSON callable in a fresh interpreter."""

    request_id: ContentIdentifier

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.request_id != self.expected_request_id():
            raise ValueError("request_id does not match process request authority")
        return self

    def expected_request_id(self) -> str:
        return content_id(
            "v72-process-request",
            self.model_dump(mode="json", exclude={"request_id"}),
        )

    @classmethod
    def identified(
        cls,
        *,
        entrypoint: str,
        payload: JsonValue,
        cwd: Path,
        authority_context: JsonValue | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        isolation: JsonProcessIsolation = "none",
        read_only_roots: tuple[Path, ...] = (),
        writable_roots: tuple[Path, ...] = (),
        maximum_cpu_seconds: float = 3_600.0,
        maximum_memory_bytes: int = 4 * 2**30,
        maximum_file_bytes: int = 4 * 2**30,
        maximum_output_entries: int = 10_000,
        maximum_capture_bytes: int = 2**20,
        code_root: Path | None = None,
    ) -> Self:
        if cwd.is_symlink():
            raise ValueError("process cwd cannot be a symlink")
        resolved_cwd = cwd.resolve(strict=True)
        if not resolved_cwd.is_dir():
            raise ValueError("process cwd must be a directory")
        resolved_read_roots: list[str] = []
        for root in read_only_roots:
            if root.is_symlink():
                raise ValueError("sandbox read-only roots cannot be symlinks")
            try:
                resolved = root.resolve(strict=True)
            except OSError as error:
                raise ValueError("sandbox read-only root does not exist") from error
            if not (
                resolved.is_dir()
                or resolved.is_file()
                or resolved
                in {
                    Path("/dev/null"),
                    Path("/dev/random"),
                    Path("/dev/urandom"),
                }
            ):
                raise ValueError("sandbox read-only roots must be regular files or directories")
            if any(ord(character) < 32 or ord(character) == 127 for character in str(resolved)):
                raise ValueError("sandbox roots cannot contain control characters")
            resolved_read_roots.append(str(resolved))
        resolved_write_roots: list[str] = []
        for root in writable_roots:
            if root.is_symlink():
                raise ValueError("sandbox writable roots cannot be symlinks")
            try:
                resolved = root.resolve(strict=True)
            except OSError as error:
                raise ValueError("sandbox writable root does not exist") from error
            if not resolved.is_dir():
                raise ValueError("sandbox writable roots must be directories")
            if any(ord(character) < 32 or ord(character) == 127 for character in str(resolved)):
                raise ValueError("sandbox roots cannot contain control characters")
            resolved_write_roots.append(str(resolved))
        if len(set(resolved_read_roots)) != len(resolved_read_roots):
            raise ValueError("sandbox read-only roots resolve to duplicates")
        if len(set(resolved_write_roots)) != len(resolved_write_roots):
            raise ValueError("sandbox writable roots resolve to duplicates")
        code_closure_sha256 = None
        resolved_code_root: Path | None = None
        if code_root is not None:
            if code_root.is_symlink():
                raise ValueError("sandbox code root cannot be a symlink")
            try:
                resolved_code_root = code_root.resolve(strict=True)
            except OSError as error:
                raise ValueError("sandbox code root does not exist") from error
            if not resolved_code_root.is_dir():
                raise ValueError("sandbox code root must be a directory")
            if str(resolved_code_root) not in resolved_read_roots:
                raise ValueError("sandbox code root must be an exact read-only root")
            if any(
                resolved_code_root.is_relative_to(Path(root))
                or Path(root).is_relative_to(resolved_code_root)
                for root in resolved_write_roots
            ):
                raise ValueError("sandbox code root must be disjoint from writable roots")
            code_closure_sha256 = tree_sha256(resolved_code_root)
        if isolation != "none" and resolved_code_root is None:
            raise ValueError("isolated execution requires an exact staged code root")
        if isolation == "none" and resolved_code_root is not None:
            raise ValueError("non-isolated execution cannot declare a staged code root")
        effective_environment = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            **dict(environment or {}),
        }
        if isolation != "none" and effective_environment.get("PYTHONPATH") != str(
            resolved_code_root
        ):
            raise ValueError("isolated execution PYTHONPATH must equal the staged code root")
        value = _JsonProcessRequestValue(
            entrypoint=entrypoint,
            payload=payload,
            authority_context=authority_context,
            cwd=str(resolved_cwd),
            environment=tuple(
                ProcessEnvironmentVariable(name=name, value=value)
                for name, value in sorted(effective_environment.items())
            ),
            timeout_seconds=timeout_seconds,
            isolation=isolation,
            read_only_roots=tuple(sorted(resolved_read_roots)),
            writable_roots=tuple(sorted(resolved_write_roots)),
            maximum_cpu_seconds=maximum_cpu_seconds,
            maximum_memory_bytes=maximum_memory_bytes,
            maximum_file_bytes=maximum_file_bytes,
            maximum_output_entries=maximum_output_entries,
            maximum_capture_bytes=maximum_capture_bytes,
            code_closure_sha256=code_closure_sha256,
            runtime_device_rdevs=tuple(
                Path(root).stat().st_rdev
                for root in sorted(resolved_read_roots)
                if Path(root) in {Path("/dev/null"), Path("/dev/random"), Path("/dev/urandom")}
            ),
            read_root_authority_sha256s=tuple(
                _mount_authority_sha256(Path(root)) for root in sorted(resolved_read_roots)
            ),
            writable_root_authority_sha256s=tuple(
                _mount_authority_sha256(Path(root)) for root in sorted(resolved_write_roots)
            ),
        )
        authority = value.model_dump(mode="json")
        return cls.model_validate_json(
            canonical_json(
                {"request_id": content_id("v72-process-request", authority), **authority}
            )
        )


class JsonProcessRequestEvidence(StrictFrozenModel):
    """Host-path-free authority for one executed JSON process request."""

    schema_version: Literal["v7.2.json-process-request-evidence.v1"] = (
        "v7.2.json-process-request-evidence.v1"
    )
    request_id: ContentIdentifier
    host_request_sha256: Sha256Digest
    entrypoint: NonEmptyString
    payload: JsonValue
    authority_context: JsonValue | None
    logical_cwd: NonEmptyString
    environment_variable_names: tuple[NonEmptyString, ...]
    timeout_seconds: PositiveTimeout
    isolation: JsonProcessIsolation
    read_mount_roles: tuple[NonEmptyString, ...]
    writable_mount_roles: tuple[NonEmptyString, ...]
    network_access: Literal[False]
    maximum_cpu_seconds: PositiveTimeout
    maximum_memory_bytes: PositiveBytes
    maximum_file_bytes: PositiveBytes
    maximum_output_entries: PositiveBytes
    maximum_capture_bytes: PositiveBytes
    code_closure_sha256: Sha256Digest | None
    mount_authorities: tuple[ProcessMountAuthority, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.request_id != content_id(
            "v72-process-request-evidence",
            self.model_dump(
                mode="json",
                exclude={"request_id", "host_request_sha256"},
            ),
        ):
            raise ValueError("published process request identity is stale")
        return self


def _sanitize_json(value: JsonValue, role: str = "value") -> JsonValue:
    if isinstance(value, str):
        replacement = f"mount:{role.replace('_', '-')}"
        return _ABSOLUTE_PATH_FRAGMENT.sub(
            lambda match: f"{match.group(1)}{replacement}",
            value,
        )
    if isinstance(value, list):
        return [_sanitize_json(item, role) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_json(item, key) for key, item in value.items()}
    return value


def _request_evidence(request: JsonProcessRequest) -> JsonProcessRequestEvidence:
    context = _sanitize_json(request.authority_context)
    role = (
        request.authority_context.get("role", "process")
        if isinstance(request.authority_context, dict)
        else "process"
    )
    read_roles = ["code", "runtime", "workspace"] if request.read_only_roots else []
    if role == "verifier":
        read_roles.append("verifier-private")
    environment = {item.name: item.value for item in request.environment}
    code_root = environment.get("PYTHONPATH")
    writable = set(request.writable_roots)
    runtime_roots = {
        str(candidate.resolve(strict=True))
        for candidate in (
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            Path(sys.prefix),
            Path(sys.executable).resolve(strict=True).parents[1],
        )
        if candidate.exists()
    }
    mount_authorities: list[ProcessMountAuthority] = []
    for root, digest in zip(
        request.read_only_roots,
        request.read_root_authority_sha256s,
        strict=True,
    ):
        path = Path(root)
        metadata = path.stat()
        if stat.S_ISCHR(metadata.st_mode):
            source_kind = "device"
        elif root in runtime_roots:
            source_kind = "runtime_directory"
        elif path.is_file():
            source_kind = "file"
        else:
            source_kind = "tree"
        if root == code_root:
            logical_role = "code"
        elif root == request.cwd or Path(request.cwd).is_relative_to(path):
            logical_role = "workspace"
        elif root in runtime_roots or source_kind == "device":
            logical_role = f"runtime:{path.name or 'root'}"
        elif role == "verifier":
            logical_role = "verifier-private"
        else:
            logical_role = "auxiliary"
        mount_authorities.append(
            ProcessMountAuthority(
                logical_role=logical_role,
                access_mode="read_write" if root in writable else "read_only",
                source_kind=source_kind,
                authority_sha256=digest,
                device_rdev=metadata.st_rdev if source_kind == "device" else None,
            )
        )
    authority_by_root = dict(
        zip(
            request.writable_roots,
            request.writable_root_authority_sha256s,
            strict=True,
        )
    )
    for root in request.writable_roots:
        if root in request.read_only_roots:
            continue
        mount_authorities.append(
            ProcessMountAuthority(
                logical_role="output",
                access_mode="read_write",
                source_kind="tree",
                authority_sha256=authority_by_root[root],
            )
        )
    ordered_mounts = tuple(
        sorted(
            mount_authorities,
            key=lambda item: (
                item.logical_role,
                item.access_mode,
                item.source_kind,
                item.authority_sha256,
            ),
        )
    )
    authority = {
        "schema_version": "v7.2.json-process-request-evidence.v1",
        "host_request_sha256": content_digest(request.model_dump(mode="json")),
        "entrypoint": request.entrypoint,
        "payload": _sanitize_json(request.payload),
        "authority_context": context,
        "logical_cwd": f"runtime:{role}",
        "environment_variable_names": [item.name for item in request.environment],
        "timeout_seconds": request.timeout_seconds,
        "isolation": request.isolation,
        "read_mount_roles": sorted(set(read_roles)),
        "writable_mount_roles": ["output"] if request.writable_roots else [],
        "network_access": request.network_access,
        "maximum_cpu_seconds": request.maximum_cpu_seconds,
        "maximum_memory_bytes": request.maximum_memory_bytes,
        "maximum_file_bytes": request.maximum_file_bytes,
        "maximum_output_entries": request.maximum_output_entries,
        "maximum_capture_bytes": request.maximum_capture_bytes,
        "code_closure_sha256": request.code_closure_sha256,
        "mount_authorities": [item.model_dump(mode="json") for item in ordered_mounts],
    }
    return JsonProcessRequestEvidence.model_validate_json(
        canonical_json(
            {
                "request_id": content_id(
                    "v72-process-request-evidence",
                    {
                        key: value
                        for key, value in authority.items()
                        if key != "host_request_sha256"
                    },
                ),
                **authority,
            }
        )
    )


class JsonProcessExecution(StrictFrozenModel):
    """Captured status and byte evidence for one JSON process request."""

    schema_version: Literal["v7.2.json-process-execution.v1"] = "v7.2.json-process-execution.v1"
    execution_id: ContentIdentifier
    request_id: ContentIdentifier
    request: JsonProcessRequestEvidence
    status: JsonProcessStatus
    process_id: int | None
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_sha256: Sha256Digest
    stderr_sha256: Sha256Digest
    raw_stdout_sha256: Sha256Digest
    raw_stderr_sha256: Sha256Digest
    response: JsonValue | None
    response_sha256: Sha256Digest | None
    raw_response_sha256: Sha256Digest | None
    error_detail: str | None
    raw_error_detail_sha256: Sha256Digest | None
    elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    cpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    peak_memory_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    output_entries: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.request_id != self.request.request_id:
            raise ValueError("process execution request authority is stale")
        if self.stdout_sha256 != sha256_bytes(self.stdout.encode("utf-8")):
            raise ValueError("stdout digest does not match captured stdout")
        if self.stderr_sha256 != sha256_bytes(self.stderr.encode("utf-8")):
            raise ValueError("stderr digest does not match captured stderr")
        if self.status == "success":
            if (
                self.exit_code != 0
                or self.response_sha256 != content_digest(self.response)
                or self.raw_response_sha256 is None
            ):
                raise ValueError("successful process evidence is incomplete")
            try:
                stdout_response = json.loads(self.stdout)
            except json.JSONDecodeError as error:
                raise ValueError("successful stdout is not canonical response JSON") from error
            if stdout_response != self.response or self.stdout != canonical_json(
                self.response
            ).decode("utf-8"):
                raise ValueError("successful stdout differs from the published response")
            if self.error_detail is not None or self.raw_error_detail_sha256 is not None:
                raise ValueError("successful process evidence cannot contain an error")
        else:
            if (
                self.response is not None
                or self.response_sha256 is not None
                or self.raw_response_sha256 is not None
            ):
                raise ValueError("failed process evidence cannot claim a JSON response")
            if not self.error_detail or self.raw_error_detail_sha256 is None:
                raise ValueError("failed process evidence requires an error detail")
        if self.status == "policy_failure" and self.exit_code != _CHILD_POLICY_FAILURE:
            raise ValueError("policy failure must use the boundary policy exit code")
        if self.status == "timeout" and self.exit_code is not None:
            raise ValueError("timeout evidence cannot claim a child exit code")
        if self.status != "infrastructure_failure" and (
            self.process_id is None or self.process_id <= 0
        ):
            raise ValueError("spawned process evidence requires a positive process ID")
        if self.execution_id != self.expected_execution_id():
            raise ValueError("execution_id does not match process execution evidence")
        return self

    def expected_execution_id(self) -> str:
        return content_id(
            "v72-process-execution",
            self.model_dump(mode="json", exclude={"execution_id"}),
        )


def _execution(
    *,
    request: JsonProcessRequest,
    status: JsonProcessStatus,
    process_id: int | None,
    exit_code: int | None,
    stdout: bytes,
    stderr: bytes,
    response: JsonValue | None,
    error_detail: str | None,
    elapsed_seconds: float = 0.0,
    cpu_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
    output_bytes: int = 0,
    output_entries: int = 0,
) -> JsonProcessExecution:
    request_evidence = _request_evidence(request)
    raw_stdout_sha256 = sha256_bytes(stdout)
    raw_stderr_sha256 = sha256_bytes(stderr)
    stdout_text = _ABSOLUTE_PATH_FRAGMENT.sub(
        lambda match: f"{match.group(1)}[absolute-path-redacted]",
        stdout.decode("utf-8", errors="replace"),
    )
    stderr_text = _ABSOLUTE_PATH_FRAGMENT.sub(
        lambda match: f"{match.group(1)}[absolute-path-redacted]",
        stderr.decode("utf-8", errors="replace"),
    )
    sanitized_response = _sanitize_json(response)
    if status == "success":
        stdout_text = canonical_json(sanitized_response).decode("utf-8")
    sanitized_error = (
        _ABSOLUTE_PATH_FRAGMENT.sub(
            lambda match: f"{match.group(1)}[absolute-path-redacted]",
            error_detail,
        )
        if error_detail is not None
        else None
    )
    authority = {
        "schema_version": "v7.2.json-process-execution.v1",
        "request_id": request_evidence.request_id,
        "request": request_evidence.model_dump(mode="json"),
        "status": status,
        "process_id": process_id,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_sha256": sha256_bytes(stdout_text.encode("utf-8")),
        "stderr_sha256": sha256_bytes(stderr_text.encode("utf-8")),
        "raw_stdout_sha256": raw_stdout_sha256,
        "raw_stderr_sha256": raw_stderr_sha256,
        "response": sanitized_response,
        "response_sha256": content_digest(sanitized_response) if status == "success" else None,
        "raw_response_sha256": content_digest(response) if status == "success" else None,
        "error_detail": sanitized_error,
        "raw_error_detail_sha256": (
            sha256_bytes(error_detail.encode("utf-8")) if error_detail is not None else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "output_bytes": output_bytes,
        "output_entries": output_entries,
    }
    return JsonProcessExecution.model_validate_json(
        canonical_json(
            {
                "execution_id": content_id("v72-process-execution", authority),
                **authority,
            }
        )
    )


def execute_json_process(request: JsonProcessRequest) -> JsonProcessExecution:
    """Run an entrypoint in a new Python interpreter and capture JSON evidence."""
    request = JsonProcessRequest.model_validate(request.model_dump(mode="python"))
    cwd = Path(request.cwd)
    try:
        cwd_is_stable = not cwd.is_symlink() and cwd.resolve(strict=True) == cwd and cwd.is_dir()
    except (OSError, ValueError):
        cwd_is_stable = False
    if not cwd_is_stable:
        return _execution(
            request=request,
            status="infrastructure_failure",
            process_id=None,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            response=None,
            error_detail="process cwd is unavailable or unsafe",
        )
    environment = {item.name: item.value for item in request.environment}
    try:
        expected_devices = dict(
            zip(
                (
                    root
                    for root in request.read_only_roots
                    if Path(root) in {Path("/dev/null"), Path("/dev/random"), Path("/dev/urandom")}
                ),
                request.runtime_device_rdevs,
                strict=True,
            )
        )
        read_roots_are_stable = all(
            not Path(root).is_symlink()
            and Path(root).resolve(strict=True) == Path(root)
            and (
                (Path(root).is_dir() or Path(root).is_file())
                if root not in expected_devices
                else stat.S_ISCHR(Path(root).stat().st_mode)
                and Path(root).stat().st_rdev == expected_devices[root]
            )
            for root in request.read_only_roots
        )
        writable_roots_are_stable = all(
            not Path(root).is_symlink()
            and Path(root).resolve(strict=True) == Path(root)
            and Path(root).is_dir()
            and any(Path(root).is_relative_to(Path(read)) for read in request.read_only_roots)
            for root in request.writable_roots
        )
        mount_authorities_are_stable = (
            tuple(_mount_authority_sha256(Path(root)) for root in request.read_only_roots)
            == request.read_root_authority_sha256s
            and tuple(_mount_authority_sha256(Path(root)) for root in request.writable_roots)
            == request.writable_root_authority_sha256s
        )
    except (OSError, ValueError):
        read_roots_are_stable = False
        writable_roots_are_stable = False
        mount_authorities_are_stable = False
    if (
        not read_roots_are_stable
        or not writable_roots_are_stable
        or not mount_authorities_are_stable
    ):
        code_authority_changed = False
        if request.isolation != "none" and environment.get("PYTHONPATH") is not None:
            try:
                code_authority_changed = (
                    tree_sha256(Path(environment["PYTHONPATH"])) != request.code_closure_sha256
                )
            except (OSError, ValueError):
                code_authority_changed = True
        return _execution(
            request=request,
            status="infrastructure_failure",
            process_id=None,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            response=None,
            error_detail=(
                "sandbox staged code authority changed before execution"
                if code_authority_changed
                else "sandbox mount roots changed after request identification"
            ),
        )
    if request.isolation != "none":
        code_root_value = environment.get("PYTHONPATH")
        try:
            code_root = Path(code_root_value or "")
            code_root_is_stable = (
                code_root_value is not None
                and code_root.is_absolute()
                and not code_root.is_symlink()
                and code_root.resolve(strict=True) == code_root
                and code_root.is_dir()
                and str(code_root) in request.read_only_roots
                and tree_sha256(code_root) == request.code_closure_sha256
            )
        except (OSError, ValueError):
            code_root_is_stable = False
        if not code_root_is_stable:
            return _execution(
                request=request,
                status="infrastructure_failure",
                process_id=None,
                exit_code=None,
                stdout=b"",
                stderr=b"",
                response=None,
                error_detail="sandbox staged code authority changed before execution",
            )
    runtime_alias_root = Path(sys.executable).resolve(strict=True).parents[1]
    runtime_configuration = Path(sys.prefix).resolve(strict=True) / "pyvenv.cfg"
    if runtime_configuration.is_file():
        for line in runtime_configuration.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if name.strip() == "home" and separator:
                runtime_alias_root = Path(value.strip()).parent
                break
    runtime_target_root = runtime_alias_root.resolve(strict=True)
    child_executable = (
        str(Path(sys.prefix).resolve(strict=True) / "bin" / "python3")
        if request.isolation != "none"
        else sys.executable
    )
    child_command = (child_executable, "-B", "-c", _CHILD_PROGRAM, request.entrypoint)
    sandbox: tempfile.TemporaryDirectory[str] | None = None
    if request.isolation == "none":
        command = child_command
    elif not _UNSHARE.is_file():
        return _execution(
            request=request,
            status="infrastructure_failure",
            process_id=None,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            response=None,
            error_detail="required unshare sandbox boundary is unavailable",
        )
    else:
        sandbox = tempfile.TemporaryDirectory(prefix="v72-json-sandbox-")
        command = (
            str(_UNSHARE),
            "--user",
            "--map-root-user",
            "--mount",
            "--net",
            "--ipc",
            "--uts",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "sh",
            "-ceu",
            _SANDBOX_PROGRAM,
            "sh",
            sandbox.name,
            request.cwd,
            str(len(request.read_only_roots)),
            *request.read_only_roots,
            str(len(request.writable_roots)),
            *request.writable_roots,
            str(runtime_alias_root),
            str(runtime_target_root),
            "--",
            *child_command,
        )

    if not _PRLIMIT.is_file():
        return _execution(
            request=request,
            status="infrastructure_failure",
            process_id=None,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            response=None,
            error_detail="required prlimit resource boundary is unavailable",
        )
    command = (
        str(_PRLIMIT),
        f"--cpu={math.ceil(request.maximum_cpu_seconds)}",
        f"--as={request.maximum_memory_bytes}",
        f"--fsize={request.maximum_file_bytes}",
        "--",
        *command,
    )

    monitor_stop = threading.Event()
    resource_limit_exceeded = threading.Event()
    observed_descendants: dict[int, float] = {}
    measurement: dict[str, float | int | str | None] = {
        "cpu_seconds": 0.0,
        "peak_memory_bytes": 0,
        "output_bytes": 0,
        "output_entries": 0,
        "limit_reason": None,
    }
    started_at = time.monotonic()

    def writable_output_usage() -> tuple[int, int, str | None]:
        output_bytes = 0
        output_entries = 0
        for root in request.writable_roots:
            try:
                pending = [Path(root)]
                while pending and output_entries <= request.maximum_output_entries:
                    directory = pending.pop()
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            output_entries += 1
                            if output_entries > request.maximum_output_entries:
                                break
                            if entry.is_symlink():
                                return output_bytes, output_entries, "unsafe writable-output entry"
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                output_bytes += entry.stat(follow_symlinks=False).st_size
                            else:
                                return output_bytes, output_entries, "unsafe writable-output entry"
            except OSError:
                return output_bytes, output_entries, "unmeasurable writable output"
        return output_bytes, output_entries, None

    def monitor_process(process_id: int) -> None:
        while not monitor_stop.wait(0.005):
            try:
                root_process = psutil.Process(process_id)
                descendants = root_process.children(recursive=True)
                processes = (root_process, *descendants)
                for child in descendants:
                    observed_descendants.setdefault(child.pid, child.create_time())
                cpu_seconds = 0.0
                memory_bytes = 0
                for child in processes:
                    with child.oneshot():
                        times = child.cpu_times()
                        cpu_seconds += times.user + times.system
                        memory_bytes += child.memory_info().rss
                measurement["cpu_seconds"] = max(float(measurement["cpu_seconds"]), cpu_seconds)
                measurement["peak_memory_bytes"] = max(
                    int(measurement["peak_memory_bytes"]), memory_bytes
                )
                output_bytes, output_entries, output_error = writable_output_usage()
                measurement["output_bytes"] = max(int(measurement["output_bytes"]), output_bytes)
                measurement["output_entries"] = max(
                    int(measurement["output_entries"]), output_entries
                )
                if output_error is not None:
                    measurement["limit_reason"] = output_error
                    resource_limit_exceeded.set()
                elif cpu_seconds > request.maximum_cpu_seconds:
                    measurement["limit_reason"] = "aggregate CPU ceiling"
                    resource_limit_exceeded.set()
                elif memory_bytes > request.maximum_memory_bytes:
                    measurement["limit_reason"] = "aggregate memory ceiling"
                    resource_limit_exceeded.set()
                elif output_bytes > request.maximum_file_bytes:
                    measurement["limit_reason"] = "aggregate writable-output ceiling"
                    resource_limit_exceeded.set()
                elif output_entries > request.maximum_output_entries:
                    measurement["limit_reason"] = "writable-output entry ceiling"
                    resource_limit_exceeded.set()
            except (psutil.Error, ProcessLookupError):
                continue

    monitor: threading.Thread | None = None

    def measured_execution(**values: object) -> JsonProcessExecution:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=1.0)
        output_bytes, output_entries, output_error = writable_output_usage()
        measurement["output_bytes"] = max(int(measurement["output_bytes"]), output_bytes)
        measurement["output_entries"] = max(int(measurement["output_entries"]), output_entries)
        if output_error is not None:
            measurement["limit_reason"] = output_error
        return _execution(
            **values,  # type: ignore[arg-type]
            elapsed_seconds=max(0.0, time.monotonic() - started_at),
            cpu_seconds=float(measurement["cpu_seconds"]),
            peak_memory_bytes=int(measurement["peak_memory_bytes"]),
            output_bytes=int(measurement["output_bytes"]),
            output_entries=int(measurement["output_entries"]),
        )

    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            return _execution(
                request=request,
                status="infrastructure_failure",
                process_id=None,
                exit_code=None,
                stdout=b"",
                stderr=b"",
                response=None,
                error_detail=f"process spawn failed: {type(error).__name__}: {error}",
            )
        monitor = threading.Thread(target=monitor_process, args=(process.pid,), daemon=True)
        monitor.start()
        capture_lock = threading.Lock()
        capture_overflow = threading.Event()
        captured_bytes = 0
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()

        def capture_stream(
            stream: object,
            buffer: bytearray,
        ) -> None:
            nonlocal captured_bytes
            while True:
                chunk = stream.read(64 * 1024)  # type: ignore[attr-defined]
                if not chunk:
                    return
                with capture_lock:
                    remaining = max(0, request.maximum_capture_bytes - captured_bytes)
                    accepted = chunk[:remaining]
                    buffer.extend(accepted)
                    captured_bytes += len(accepted)
                    if len(accepted) != len(chunk):
                        capture_overflow.set()

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_reader = threading.Thread(
            target=capture_stream,
            args=(process.stdout, stdout_buffer),
            daemon=True,
        )
        stderr_reader = threading.Thread(
            target=capture_stream,
            args=(process.stderr, stderr_buffer),
            daemon=True,
        )
        stdout_reader.start()
        stderr_reader.start()
        try:
            process.stdin.write(canonical_json(request.payload))
            process.stdin.close()
        except BrokenPipeError:
            pass

        timed_out = False
        resource_exceeded = False
        deadline = started_at + request.timeout_seconds
        while process.poll() is None:
            if capture_overflow.is_set():
                resource_exceeded = True
                measurement["limit_reason"] = "combined stdout and stderr capture ceiling"
                break
            if resource_limit_exceeded.is_set():
                resource_exceeded = True
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.005)
        if timed_out or resource_exceeded:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
        process.wait()
        for descendant_id, create_time in tuple(observed_descendants.items()):
            try:
                descendant = psutil.Process(descendant_id)
                if descendant.create_time() == create_time and descendant.is_running():
                    descendant.kill()
            except (psutil.Error, ProcessLookupError):
                continue
        stdout_reader.join(timeout=1.0)
        stderr_reader.join(timeout=1.0)
        stdout = bytes(stdout_buffer)
        stderr = bytes(stderr_buffer)
        resource_exceeded = resource_exceeded or capture_overflow.is_set()
        final_output_bytes, final_output_entries, final_output_error = writable_output_usage()
        measurement["output_bytes"] = max(int(measurement["output_bytes"]), final_output_bytes)
        measurement["output_entries"] = max(
            int(measurement["output_entries"]), final_output_entries
        )
        if final_output_error is not None:
            measurement["limit_reason"] = final_output_error
            resource_exceeded = True
        if final_output_bytes > request.maximum_file_bytes:
            measurement["limit_reason"] = "aggregate writable-output ceiling"
            resource_exceeded = True
        if final_output_entries > request.maximum_output_entries:
            measurement["limit_reason"] = "writable-output entry ceiling"
            resource_exceeded = True
        resource_signals = {
            -signal.SIGXCPU,
            -signal.SIGXFSZ,
            128 + signal.SIGXCPU,
            128 + signal.SIGXFSZ,
            _CHILD_RESOURCE_FAILURE,
        }
        if process.returncode in resource_signals:
            measurement["limit_reason"] = (
                "address-space memory ceiling"
                if process.returncode == _CHILD_RESOURCE_FAILURE
                else "kernel resource limit"
            )
            resource_exceeded = True

        if resource_exceeded:
            return measured_execution(
                request=request,
                status="resource_failure",
                process_id=process.pid,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                response=None,
                error_detail=f"process exceeded the {measurement['limit_reason']}",
            )
        if timed_out:
            return measured_execution(
                request=request,
                status="timeout",
                process_id=process.pid,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                response=None,
                error_detail=f"process exceeded {request.timeout_seconds:g} seconds",
            )

        if process.returncode == 0:
            try:
                response = json.loads(stdout)
                canonical_json(response)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                return measured_execution(
                    request=request,
                    status="infrastructure_failure",
                    process_id=process.pid,
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    response=None,
                    error_detail=f"child returned invalid JSON: {type(error).__name__}: {error}",
                )
            return measured_execution(
                request=request,
                status="success",
                process_id=process.pid,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                response=response,
                error_detail=None,
            )
        if process.returncode == _CHILD_POLICY_FAILURE:
            status: JsonProcessStatus = "policy_failure"
            detail = "entrypoint raised while processing the JSON request"
        else:
            status = "infrastructure_failure"
            detail = (
                "JSON boundary setup failed"
                if process.returncode == _CHILD_INFRASTRUCTURE_FAILURE
                else f"child exited with undeclared code {process.returncode}"
            )
        return measured_execution(
            request=request,
            status=status,
            process_id=process.pid,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            response=None,
            error_detail=detail,
        )
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=1.0)
        if sandbox is not None:
            sandbox.cleanup()


__all__ = [
    "JsonProcessExecution",
    "JsonProcessIsolation",
    "JsonProcessRequest",
    "JsonProcessStatus",
    "ProcessEnvironmentVariable",
    "execute_json_process",
]
