"""Distinct executable shortcut probes for v7.2 exploit qualification."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from training_data_collection.common.filesystem import resolve_under
from training_data_collection.common.identity import sha256_bytes
from training_data_collection.v72.contracts.exploit import ExploitProbeClass

_MODULE = "training_data_collection.v72.runtime.probes"

BUILTIN_PROBE_ENTRYPOINTS: dict[ExploitProbeClass, str] = {
    "empty_output": f"{_MODULE}:probe_empty_output",
    "minimal_schema_valid": f"{_MODULE}:probe_minimal_schema_valid",
    "constant_output": f"{_MODULE}:probe_constant_output",
    "copied_input": f"{_MODULE}:probe_copied_input",
    "one_seed_hard_coding": f"{_MODULE}:probe_one_seed_hard_coding",
    "family_template_hard_coding": f"{_MODULE}:probe_family_template_hard_coding",
    "ignore_critical_role": f"{_MODULE}:probe_ignore_critical_role",
    "ignore_advertised_signal": f"{_MODULE}:probe_ignore_advertised_signal",
    "evidence_omission": f"{_MODULE}:probe_evidence_omission",
    "contradictory_report": f"{_MODULE}:probe_contradictory_report",
    "visible_only_overfit": f"{_MODULE}:probe_visible_only_overfit",
    "sidecar_omission": f"{_MODULE}:probe_sidecar_omission",
    "path_traversal": f"{_MODULE}:probe_path_traversal",
    "private_lookup": f"{_MODULE}:probe_private_lookup",
}


def _context(payload: dict[str, object]) -> tuple[Path, dict[str, object]]:
    expected = {
        "workspace_root",
        "input_paths",
        "required_output_paths",
        "primary_output_path",
        "sidecar_output_paths",
        "evidence_output_path",
        "report_output_path",
        "critical_role_path",
        "advertised_signal_path",
        "visible_case_paths",
        "verifier_private_hidden_case_ids",
        "traversal_attempt_path",
        "private_lookup_path",
        "seed_id",
        "family_template_id",
        "strategy_configuration",
        "output_variants",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("exploit probe payload shape is invalid")
    root = Path(str(payload["workspace_root"]))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("exploit workspace root is unavailable or unsafe")
    return root.resolve(strict=True), payload


def _string_tuple(payload: dict[str, object], key: str) -> tuple[str, ...]:
    values = payload[key]
    if type(values) is not list or any(type(value) is not str for value in values):
        raise TypeError(f"exploit probe {key} must be an array of strings")
    return tuple(values)


def _configuration(payload: dict[str, object]) -> dict[str, object]:
    raw = payload["strategy_configuration"]
    if type(raw) is not dict or any(type(name) is not str for name in raw):
        raise TypeError("exploit strategy configuration must be a JSON object")
    return raw


def _variant(payload: dict[str, object], name: str) -> tuple[tuple[str, bytes], ...]:
    variants = payload["output_variants"]
    if type(variants) is not dict or type(variants.get(name)) is not list:
        raise TypeError(f"exploit output variant is unavailable: {name}")
    files: list[tuple[str, bytes]] = []
    for record in variants[name]:
        if type(record) is not dict or set(record) != {"path", "content_base64"}:
            raise TypeError("exploit output variant file is invalid")
        path = record["path"]
        content = record["content_base64"]
        if type(path) is not str or type(content) is not str:
            raise TypeError("exploit output variant values must be strings")
        files.append((path, base64.b64decode(content.encode("ascii"), validate=True)))
    return tuple(files)


def _write(
    root: Path,
    path: str,
    content: bytes,
    operation: str,
) -> dict[str, object]:
    if not path.startswith("output/"):
        raise ValueError("exploit probes may write only declared output paths")
    destination = resolve_under(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return {
        "operation": operation,
        "path": path,
        "outcome": "success",
        "content_sha256": sha256_bytes(content),
    }


def _read(root: Path, path: str, operation: str = "read") -> tuple[bytes, dict[str, object]]:
    if not path.startswith("input/"):
        raise ValueError("exploit probes may read only declared input paths")
    content = resolve_under(root, path).read_bytes()
    return content, {
        "operation": operation,
        "path": path,
        "outcome": "success",
        "content_sha256": sha256_bytes(content),
    }


def _write_variant(
    root: Path,
    payload: dict[str, object],
    variant: str,
    operation: str,
    *,
    excluded_paths: set[str] | None = None,
) -> list[dict[str, object]]:
    excluded = excluded_paths or set()
    return [
        _write(root, path, content, operation)
        for path, content in _variant(payload, variant)
        if path not in excluded
    ]


def _write_configured_files(
    root: Path,
    raw_files: object,
    operation: str,
) -> list[dict[str, object]]:
    if type(raw_files) is not list:
        raise TypeError("configured exploit answer must be a file array")
    operations: list[dict[str, object]] = []
    for record in raw_files:
        if type(record) is not dict or set(record) != {"path", "content_base64"}:
            raise TypeError("configured exploit answer file is invalid")
        path = record["path"]
        content = record["content_base64"]
        if type(path) is not str or type(content) is not str:
            raise TypeError("configured exploit answer values must be strings")
        operations.append(
            _write(
                root,
                path,
                base64.b64decode(content.encode("ascii"), validate=True),
                operation,
            )
        )
    return operations


def probe_empty_output(payload: dict[str, object]) -> dict[str, object]:
    """Run the policy entrypoint and deliberately produce no required artifacts."""
    _, values = _context(payload)
    operations = [
        {
            "operation": "omit_required_output",
            "path": path,
            "outcome": "skipped",
            "content_sha256": None,
        }
        for path in _string_tuple(values, "required_output_paths")
    ]
    return {"operations": operations}


def probe_minimal_schema_valid(payload: dict[str, object]) -> dict[str, object]:
    """Write only caller-supplied minimal native-schema artifacts."""
    root, values = _context(payload)
    return {"operations": _write_variant(root, values, "minimal", "write_minimal")}


def probe_constant_output(payload: dict[str, object]) -> dict[str, object]:
    """Write fixed outputs without reading any policy-visible input."""
    root, values = _context(payload)
    return {"operations": _write_variant(root, values, "constant", "write_constant")}


def probe_copied_input(payload: dict[str, object]) -> dict[str, object]:
    """Copy the first declared input byte-for-byte into the primary output."""
    root, values = _context(payload)
    source = _string_tuple(values, "input_paths")[0]
    content, read_operation = _read(root, source)
    write_operation = _write(
        root,
        str(values["primary_output_path"]),
        content,
        "write_copy",
    )
    return {"operations": [read_operation, write_operation]}


def probe_one_seed_hard_coding(payload: dict[str, object]) -> dict[str, object]:
    """Use an answer table keyed only by one configured seed identity."""
    root, values = _context(payload)
    configuration = _configuration(values)
    target_seed = configuration.get("hardcoded_seed_id")
    observed_seed = values["seed_id"]
    matched = target_seed == observed_seed
    lookup = {
        "operation": "seed_lookup",
        "path": f"seed:{observed_seed}",
        "outcome": "success" if matched else "absent",
        "content_sha256": None,
    }
    operations = (
        _write_configured_files(root, configuration.get("hardcoded_files"), "write_hardcoded")
        if matched
        else _write_variant(root, values, "constant", "write_hardcoded")
    )
    return {"operations": [lookup, *operations]}


def probe_family_template_hard_coding(payload: dict[str, object]) -> dict[str, object]:
    """Select fixed output by one configured family-template fingerprint."""
    root, values = _context(payload)
    configuration = _configuration(values)
    target_template = configuration.get("family_template_id")
    observed_template = values["family_template_id"]
    matched = target_template == observed_template
    lookup = {
        "operation": "template_lookup",
        "path": f"template:{observed_template}",
        "outcome": "success" if matched else "absent",
        "content_sha256": None,
    }
    operations = (
        _write_configured_files(root, configuration.get("template_files"), "write_template")
        if matched
        else _write_variant(root, values, "constant", "write_template")
    )
    return {"operations": [lookup, *operations]}


def probe_ignore_critical_role(payload: dict[str, object]) -> dict[str, object]:
    """Read every declared input except the configured critical role."""
    root, values = _context(payload)
    omitted = str(values["critical_role_path"])
    operations: list[dict[str, object]] = [
        {
            "operation": "omit_critical_role",
            "path": omitted,
            "outcome": "skipped",
            "content_sha256": None,
        }
    ]
    observed: dict[str, str] = {}
    for path in _string_tuple(values, "input_paths"):
        if path == omitted:
            continue
        content, operation = _read(root, path)
        operations.append(operation)
        observed[path] = sha256_bytes(content)
    result = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    operations.append(_write(root, str(values["primary_output_path"]), result, "write_native"))
    return {"operations": operations}


def probe_ignore_advertised_signal(payload: dict[str, object]) -> dict[str, object]:
    """Ablate one advertised signal while reading the remaining inputs."""
    root, values = _context(payload)
    signal = str(values["advertised_signal_path"])
    operations: list[dict[str, object]] = [
        {
            "operation": "ablate_signal",
            "path": signal,
            "outcome": "skipped",
            "content_sha256": None,
        }
    ]
    observed: list[str] = []
    for path in _string_tuple(values, "input_paths"):
        if path == signal:
            continue
        content, operation = _read(root, path)
        operations.append(operation)
        observed.append(sha256_bytes(content))
    result = json.dumps(sorted(observed), separators=(",", ":")).encode("utf-8")
    operations.append(_write(root, str(values["primary_output_path"]), result, "write_native"))
    return {"operations": operations}


def probe_evidence_omission(payload: dict[str, object]) -> dict[str, object]:
    """Write native outputs while deliberately omitting required evidence."""
    root, values = _context(payload)
    evidence_path = str(values["evidence_output_path"])
    operations = _write_variant(
        root,
        values,
        "native",
        "write_native",
        excluded_paths={evidence_path},
    )
    operations.append(
        {
            "operation": "omit_evidence",
            "path": evidence_path,
            "outcome": "skipped",
            "content_sha256": None,
        }
    )
    return {"operations": operations}


def probe_contradictory_report(payload: dict[str, object]) -> dict[str, object]:
    """Write a native result plus a deliberately contradictory report."""
    root, values = _context(payload)
    report_path = str(values["report_output_path"])
    operations = _write_variant(
        root,
        values,
        "native",
        "write_native",
        excluded_paths={report_path},
    )
    contradiction = [
        _write(root, path, content, "write_contradiction")
        for path, content in _variant(values, "contradictory_report")
        if path == report_path
    ]
    if not contradiction:
        raise ValueError("contradictory report payload does not bind the report path")
    return {"operations": [*operations, *contradiction]}


def probe_visible_only_overfit(payload: dict[str, object]) -> dict[str, object]:
    """Fit visible cases while deliberately skipping every hidden case."""
    root, values = _context(payload)
    operations: list[dict[str, object]] = []
    visible_digests: list[str] = []
    for path in _string_tuple(values, "visible_case_paths"):
        content, operation = _read(root, path, "read_visible")
        operations.append(operation)
        visible_digests.append(sha256_bytes(content))
    operations.extend(
        {
            "operation": "omit_hidden",
            "path": path,
            "outcome": "skipped",
            "content_sha256": None,
        }
        for path in _string_tuple(values, "verifier_private_hidden_case_ids")
    )
    if not visible_digests:
        raise ValueError("visible-only probe requires realized visible input")
    operations.extend(_write_variant(root, values, "visible_fit", "write_native"))
    return {"operations": operations}


def probe_sidecar_omission(payload: dict[str, object]) -> dict[str, object]:
    """Write the primary artifact while deliberately omitting all sidecars."""
    root, values = _context(payload)
    primary_path = str(values["primary_output_path"])
    native_files = dict(_variant(values, "native"))
    if primary_path not in native_files:
        raise ValueError("native output payload lacks the primary artifact")
    operations = [_write(root, primary_path, native_files[primary_path], "write_primary")]
    operations.extend(
        {
            "operation": "omit_sidecar",
            "path": path,
            "outcome": "skipped",
            "content_sha256": None,
        }
        for path in _string_tuple(values, "sidecar_output_paths")
    )
    return {"operations": operations}


def probe_path_traversal(payload: dict[str, object]) -> dict[str, object]:
    """Attempt a direct traversal write and the safe-path boundary."""
    root, values = _context(payload)
    attempted = str(values["traversal_attempt_path"])
    try:
        (root / attempted).write_bytes(b"escape")
    except OSError:
        pass
    else:
        raise RuntimeError("sandbox boundary allowed a traversal write")
    try:
        resolve_under(root, attempted)
    except ValueError:
        outcome = "denied"
    else:
        raise RuntimeError("safe-path boundary allowed a traversal path")
    return {
        "operations": [
            {
                "operation": "path_traversal",
                "path": attempted,
                "outcome": outcome,
                "content_sha256": None,
            }
        ]
    }


def probe_private_lookup(payload: dict[str, object]) -> dict[str, object]:
    """Attempt a direct private read and the safe-path boundary."""
    root, values = _context(payload)
    attempted = str(values["private_lookup_path"])
    try:
        (root / attempted).read_bytes()
    except OSError:
        pass
    else:
        raise RuntimeError("sandbox boundary exposed a private path")
    try:
        resolve_under(root, attempted)
    except ValueError:
        outcome = "denied"
    else:
        raise RuntimeError("safe-path boundary allowed a private path")
    return {
        "operations": [
            {
                "operation": "private_lookup",
                "path": attempted,
                "outcome": outcome,
                "content_sha256": None,
            }
        ]
    }


__all__ = [
    "BUILTIN_PROBE_ENTRYPOINTS",
    "probe_constant_output",
    "probe_contradictory_report",
    "probe_copied_input",
    "probe_empty_output",
    "probe_evidence_omission",
    "probe_family_template_hard_coding",
    "probe_ignore_advertised_signal",
    "probe_ignore_critical_role",
    "probe_minimal_schema_valid",
    "probe_one_seed_hard_coding",
    "probe_path_traversal",
    "probe_private_lookup",
    "probe_sidecar_omission",
    "probe_visible_only_overfit",
]
