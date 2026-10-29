"""Static and runtime dependency-closure qualification for implementations."""

from __future__ import annotations

import ast
import contextlib
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal, Self

from pydantic import field_validator, model_validator

from training_data_collection.common.filesystem import safe_relative_path
from training_data_collection.common.identity import (
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
from training_data_collection.v72.contracts.implementation import (
    ImplementationAuthority,
    ImplementationAuthorityKind,
)

_REQUIRED_AUTHORITY_KINDS = {
    "materializer",
    "reference",
    "alternative",
    "verifier",
}
_MODULE_ID = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_REVIEWED_INFRASTRUCTURE_PREFIXES = (
    "training_data_collection.common",
    "training_data_collection.v72.contracts",
    "training_data_collection.v72.generation",
)
_REVIEWED_INFRASTRUCTURE_MODULES = {
    "training_data_collection",
    "training_data_collection.v72",
    "training_data_collection.v72.families.base",
}
_DYNAMIC_IMPORT_INFRASTRUCTURE_MODULES = {
    "training_data_collection.v72.families.registry",
    "training_data_collection.v72.qualification.family_revision",
    "training_data_collection.v72.runtime.execution",
}
_RUNTIME_DISCOVERY_PROGRAM = r"""
import contextlib
import importlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
module_name, qualname = sys.argv[2].split(":", 1)
with contextlib.redirect_stdout(sys.stderr):
    module = importlib.import_module(module_name)
    target = module
    for part in qualname.split("."):
        target = getattr(target, part)
if not callable(target):
    raise TypeError("implementation entrypoint is not callable")
records = {}
for name, loaded in tuple(sys.modules.items()):
    raw_path = getattr(loaded, "__file__", None)
    if not raw_path:
        continue
    path = pathlib.Path(raw_path)
    if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
        path = path.with_suffix(".py")
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError):
        continue
    if resolved.suffix == ".py":
        records[name] = relative.as_posix()
sys.stdout.write(json.dumps(records, sort_keys=True, separators=(",", ":")))
"""


class DependencyModuleRecord(StrictFrozenModel):
    """One source file in a statically or dynamically discovered closure."""

    module_id: str
    relative_path: str
    source_sha256: Sha256Digest
    discovered_by: tuple[Literal["ast", "runtime"], ...]

    @field_validator("module_id")
    @classmethod
    def validate_module_id(cls, value: str) -> str:
        if _MODULE_ID.fullmatch(value) is None:
            raise ValueError("dependency module ID is not an absolute Python module name")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if safe_relative_path(value).as_posix() != value or not value.endswith(".py"):
            raise ValueError("dependency source path must be a canonical relative Python file")
        return value

    @field_validator("discovered_by")
    @classmethod
    def validate_discovery_kinds(
        cls,
        value: tuple[Literal["ast", "runtime"], ...],
    ) -> tuple[Literal["ast", "runtime"], ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("dependency discovery kinds must be sorted and unique")
        return value


class ImplementationDependencyClosure(StrictFrozenModel):
    """Content-bound union of AST and fresh-runtime module discovery."""

    schema_version: Literal["v7.2.implementation-closure.v1"] = "v7.2.implementation-closure.v1"
    closure_id: ContentIdentifier
    entrypoint: str
    entrypoint_module_id: str
    source_tree_sha256: Sha256Digest
    dependency_closure_sha256: Sha256Digest
    modules: tuple[DependencyModuleRecord, ...]
    static_module_ids: tuple[str, ...]
    runtime_module_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_closure(self) -> Self:
        module_ids = tuple(item.module_id for item in self.modules)
        if not module_ids or module_ids != tuple(sorted(set(module_ids))):
            raise ValueError("dependency closure modules must be nonempty, sorted, and unique")
        if self.entrypoint_module_id not in module_ids:
            raise ValueError("dependency closure does not contain its entrypoint module")
        for values, label in (
            (self.static_module_ids, "static"),
            (self.runtime_module_ids, "runtime"),
        ):
            if values != tuple(sorted(set(values))) or not set(values) <= set(module_ids):
                raise ValueError(f"{label} module identities are not a canonical closure subset")
        expected_source = content_digest(
            [
                {
                    "module_id": item.module_id,
                    "relative_path": item.relative_path,
                    "source_sha256": item.source_sha256,
                }
                for item in self.modules
                if item.module_id == self.entrypoint_module_id
            ]
        )
        if self.source_tree_sha256 != expected_source:
            raise ValueError("source tree digest does not bind the entrypoint source")
        expected_closure = content_digest([item.model_dump(mode="json") for item in self.modules])
        if self.dependency_closure_sha256 != expected_closure:
            raise ValueError("dependency closure digest does not bind every module source")
        authority = self.model_dump(mode="json", exclude={"closure_id"})
        if self.closure_id != content_id("v72-implementation-closure", authority):
            raise ValueError("closure_id does not match dependency closure authority")
        return self


class ImplementationIndependenceEvidence(StrictFrozenModel):
    """Successful four-role source and dependency independence evidence."""

    schema_version: Literal["v7.2.implementation-independence.v1"] = (
        "v7.2.implementation-independence.v1"
    )
    evidence_id: ContentIdentifier
    authority_ids: tuple[ContentIdentifier, ...]
    closure_ids: tuple[ContentIdentifier, ...]
    shared_module_ids: tuple[str, ...]
    qualified: Literal[True]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.authority_ids != tuple(sorted(set(self.authority_ids))):
            raise ValueError("implementation authority IDs must be sorted and unique")
        if self.closure_ids != tuple(sorted(set(self.closure_ids))):
            raise ValueError("implementation closure IDs must be sorted and unique")
        if self.shared_module_ids != tuple(sorted(set(self.shared_module_ids))):
            raise ValueError("shared module IDs must be sorted and unique")
        authority = self.model_dump(mode="json", exclude={"evidence_id"})
        if self.evidence_id != content_id("v72-implementation-independence", authority):
            raise ValueError("evidence_id does not match independence evidence")
        return self


def _module_path(source_root: Path, module_id: str) -> Path | None:
    parts = module_id.split(".")
    candidates = (
        source_root.joinpath(*parts).with_suffix(".py"),
        source_root.joinpath(*parts, "__init__.py"),
    )
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        resolved = candidate.resolve(strict=True)
        with contextlib.suppress(ValueError):
            resolved.relative_to(source_root)
            return resolved
    return None


def _relative_import_base(module_id: str, path: Path, level: int) -> list[str]:
    package_parts = module_id.split(".")
    if path.name != "__init__.py":
        package_parts.pop()
    if level > len(package_parts) + 1:
        return []
    return package_parts[: len(package_parts) - level + 1]


def _import_candidates(module_id: str, path: Path, node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if node.level:
        base_parts = _relative_import_base(module_id, path, node.level)
        if not base_parts and node.level:
            return set()
        if node.module:
            base_parts.extend(node.module.split("."))
        base = ".".join(base_parts)
    else:
        base = node.module or ""
    candidates = {base} if base else set()
    candidates.update(
        f"{base}.{alias.name}" if base else alias.name for alias in node.names if alias.name != "*"
    )
    return candidates


def _is_type_checking_import_block(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
        and not node.orelse
        and bool(node.body)
        and all(isinstance(statement, (ast.Import, ast.ImportFrom)) for statement in node.body)
    )


def _reject_dynamic_imports(tree: ast.AST, path: Path, module_id: str) -> None:
    if module_id in _DYNAMIC_IMPORT_INFRASTRUCTURE_MODULES:
        return
    builtins_aliases = {"builtins", "__builtins__"}
    forbidden_builtin_names = {"__import__", "eval", "exec"}
    getattr_aliases = {"getattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "importlib" for alias in node.names):
                raise ValueError(f"dynamic imports are forbidden in role closure source: {path}")
            builtins_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "builtins"
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "builtins":
                getattr_aliases.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "getattr"
                )
            if node.module == "importlib" or (
                node.module == "builtins"
                and any(alias.name in forbidden_builtin_names for alias in node.names)
            ):
                raise ValueError(f"dynamic imports are forbidden in role closure source: {path}")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in forbidden_builtin_names | {"importlib"}
        ):
            raise ValueError(f"dynamic imports are forbidden in role closure source: {path}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in builtins_aliases
            and node.attr in forbidden_builtin_names
        ):
            raise ValueError(f"dynamic imports are forbidden in role closure source: {path}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in getattr_aliases
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in forbidden_builtin_names | {"import_module"}
        ):
            raise ValueError(f"dynamic imports are forbidden in role closure source: {path}")


def _static_module_ids(entrypoint_module: str, source_root: Path) -> set[str]:
    pending = deque([entrypoint_module])
    discovered: set[str] = set()
    while pending:
        module_id = pending.popleft()
        if module_id in discovered:
            continue
        path = _module_path(source_root, module_id)
        if path is None:
            continue
        discovered.add(module_id)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _reject_dynamic_imports(tree, path, module_id)
        ignored_imports = {
            id(child)
            for statement in tree.body
            if _is_type_checking_import_block(statement)
            for child in ast.walk(statement)
            if isinstance(child, (ast.Import, ast.ImportFrom))
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in ignored_imports:
                continue
            for candidate in sorted(_import_candidates(module_id, path, node)):
                if candidate not in discovered and _module_path(source_root, candidate):
                    pending.append(candidate)
    if entrypoint_module not in discovered:
        raise ValueError("implementation entrypoint module is outside the source root")
    return discovered


def _runtime_modules(entrypoint: str, source_root: Path) -> dict[str, str]:
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    try:
        result = subprocess.run(
            (
                sys.executable,
                "-B",
                "-c",
                _RUNTIME_DISCOVERY_PROGRAM,
                str(source_root),
                entrypoint,
            ),
            cwd=source_root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"runtime dependency discovery failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"runtime dependency discovery failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime dependency discovery returned invalid JSON") from error
    if type(payload) is not dict or any(
        type(module_id) is not str or type(path) is not str for module_id, path in payload.items()
    ):
        raise ValueError("runtime dependency discovery returned an invalid module map")
    return payload


def build_dependency_closure(
    entrypoint: str,
    source_root: Path,
) -> ImplementationDependencyClosure:
    """Hash the reproducible union of static and fresh-runtime local dependencies."""
    if ":" not in entrypoint:
        raise ValueError("implementation entrypoint must contain a qualified-name separator")
    module_id, _ = entrypoint.split(":", maxsplit=1)
    if _MODULE_ID.fullmatch(module_id) is None:
        raise ValueError("implementation entrypoint module is invalid")
    root = source_root.resolve(strict=True)
    if source_root.is_symlink() or not root.is_dir():
        raise ValueError("implementation source root must be a real directory")
    static_ids = _static_module_ids(module_id, root)
    runtime_paths = _runtime_modules(entrypoint, root)
    runtime_ids = set(runtime_paths)
    records: list[DependencyModuleRecord] = []
    for dependency_id in sorted(static_ids | runtime_ids):
        path = _module_path(root, dependency_id)
        if path is None:
            raise ValueError(f"discovered dependency has no stable Python source: {dependency_id}")
        relative_path = path.relative_to(root).as_posix()
        runtime_path = runtime_paths.get(dependency_id)
        if runtime_path is not None and runtime_path != relative_path:
            raise ValueError(f"runtime dependency path disagrees with source root: {dependency_id}")
        records.append(
            DependencyModuleRecord(
                module_id=dependency_id,
                relative_path=relative_path,
                source_sha256=sha256_bytes(path.read_bytes()),
                discovered_by=tuple(
                    discovery
                    for discovery, present in (
                        ("ast", dependency_id in static_ids),
                        ("runtime", dependency_id in runtime_ids),
                    )
                    if present
                ),
            )
        )
    source_tree_sha256 = content_digest(
        [
            {
                "module_id": item.module_id,
                "relative_path": item.relative_path,
                "source_sha256": item.source_sha256,
            }
            for item in records
            if item.module_id == module_id
        ]
    )
    dependency_closure_sha256 = content_digest([item.model_dump(mode="json") for item in records])
    authority = {
        "schema_version": "v7.2.implementation-closure.v1",
        "entrypoint": entrypoint,
        "entrypoint_module_id": module_id,
        "source_tree_sha256": source_tree_sha256,
        "dependency_closure_sha256": dependency_closure_sha256,
        "modules": [item.model_dump(mode="json") for item in records],
        "static_module_ids": sorted(static_ids),
        "runtime_module_ids": sorted(runtime_ids),
    }
    return ImplementationDependencyClosure.model_validate_json(
        canonical_json(
            {
                "closure_id": content_id("v72-implementation-closure", authority),
                **authority,
            }
        )
    )


def build_implementation_authority(
    *,
    authority_kind: ImplementationAuthorityKind,
    entrypoint: str,
    source_root: Path,
    allowed_shared_module_ids: tuple[str, ...],
    algorithm_description: str,
) -> ImplementationAuthority:
    """Build an authority from measured source and dependency closure evidence."""
    if not algorithm_description.strip():
        raise ValueError("algorithm description must not be empty")
    closure = build_dependency_closure(entrypoint, source_root)
    return ImplementationAuthority.identified(
        authority_kind=authority_kind,
        source_tree_sha256=closure.source_tree_sha256,
        entrypoint=entrypoint,
        dependency_closure_sha256=closure.dependency_closure_sha256,
        allowed_shared_module_ids=allowed_shared_module_ids,
        algorithm_description_digest=content_digest(
            {"algorithm_description": algorithm_description}
        ),
    )


_REVIEWED_SHARED_ROLE_INFRASTRUCTURE = (
    "training_data_collection",
    "training_data_collection.common",
    "training_data_collection.common.filesystem",
    "training_data_collection.common.identity",
    "training_data_collection.v72",
    "training_data_collection.v72.contracts",
    "training_data_collection.v72.contracts.corruption",
    "training_data_collection.v72.contracts.identity",
    "training_data_collection.v72.contracts.implementation",
    "training_data_collection.v72.contracts.world",
    "training_data_collection.v72.families",
    "training_data_collection.v72.generation",
    "training_data_collection.v72.generation.corruption",
    "training_data_collection.v72.generation.world",
)


def reviewed_shared_module_allowlist(family_package: str) -> tuple[str, ...]:
    """Return the fixed reviewed structural allowlist for one role package."""
    if family_package == "tests.v72.packaging_roles":
        return ("tests.v72", "tests.v72.packaging_roles")
    if family_package.startswith("training_data_collection.v72.families.cm."):
        family_group = "training_data_collection.v72.families.cm"
    elif family_package.startswith("training_data_collection.v72.families.ml."):
        family_group = "training_data_collection.v72.families.ml"
    elif family_package.startswith("training_data_collection.v72.families.sre."):
        family_group = "training_data_collection.v72.families.sre"
    elif family_package.startswith("training_data_collection.v72.families.physical."):
        family_group = "training_data_collection.v72.families.physical"
    elif family_package.startswith("training_data_collection.v72.families.code."):
        family_group = "training_data_collection.v72.families.code"
    elif family_package.startswith("training_data_collection.v72.families.optimization."):
        family_group = "training_data_collection.v72.families.optimization"
    elif family_package.startswith("training_data_collection.v72.families.mathematical."):
        family_group = "training_data_collection.v72.families.mathematical"
    elif "." not in family_package:
        return (family_package,)
    else:
        raise ValueError("family package has no reviewed shared-module policy")
    return tuple(sorted((*_REVIEWED_SHARED_ROLE_INFRASTRUCTURE, family_group, family_package)))


def _has_shared_semantic_authority(module_id: str, path: Path) -> bool:
    if module_id in _REVIEWED_INFRASTRUCTURE_MODULES or module_id.startswith(
        tuple(f"{prefix}." for prefix in _REVIEWED_INFRASTRUCTURE_PREFIXES)
    ):
        return False
    if path.name != "__init__.py":
        return True
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if any(isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)):
        return True

    def is_structural_lazy_export(node: ast.FunctionDef) -> bool:
        if (
            node.name != "__getattr__"
            or node.decorator_list
            or len(node.args.args) != 1
            or node.args.args[0].arg != "name"
            or node.args.vararg is not None
            or node.args.kwarg is not None
            or node.args.kwonlyargs
        ):
            return False
        allowed = (
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
            ast.If,
            ast.Compare,
            ast.Name,
            ast.Load,
            ast.NotIn,
            ast.Raise,
            ast.Call,
            ast.ImportFrom,
            ast.alias,
            ast.Return,
            ast.Subscript,
            ast.Dict,
            ast.Constant,
        )
        if any(not isinstance(child, allowed) for child in ast.walk(node)):
            return False
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        if any(
            not isinstance(call.func, ast.Name) or call.func.id != "AttributeError"
            for call in calls
        ):
            return False
        return_values = [child.value for child in node.body if isinstance(child, ast.Return)]
        if len(return_values) != 1 or not isinstance(return_values[0], ast.Subscript):
            return False
        mapping = return_values[0].value
        selector = return_values[0].slice
        return (
            isinstance(mapping, ast.Dict)
            and isinstance(selector, ast.Name)
            and selector.id == "name"
            and all(
                isinstance(key, ast.Constant) and isinstance(key.value, str) for key in mapping.keys
            )
            and all(isinstance(value, ast.Name) for value in mapping.values)
        )

    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _is_type_checking_import_block(node):
            continue
        if isinstance(node, ast.FunctionDef) and is_structural_lazy_export(node):
            continue
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
                or node.targets[0].id != "__all__"
                or not isinstance(node.value, (ast.List, ast.Tuple))
                or any(
                    not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                    for item in node.value.elts
                )
            ):
                return True
            continue
        if isinstance(node, ast.AnnAssign):
            if (
                not isinstance(node.target, ast.Name)
                or node.target.id != "__all__"
                or not isinstance(node.value, (ast.List, ast.Tuple))
                or any(
                    not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                    for item in node.value.elts
                )
            ):
                return True
            continue
        return True
    return False


def audit_implementation_independence(
    authorities: tuple[ImplementationAuthority, ...],
    source_root: Path,
    *,
    reviewed_shared_module_ids: tuple[str, ...],
) -> ImplementationIndependenceEvidence:
    """Require distinct four-role algorithms with closed, audited dependencies."""
    by_kind = {authority.authority_kind: authority for authority in authorities}
    if len(by_kind) != len(authorities) or set(by_kind) != _REQUIRED_AUTHORITY_KINDS:
        raise ValueError("independence audit requires exactly one authority for all four roles")
    closures: dict[str, ImplementationDependencyClosure] = {}
    for kind, authority in by_kind.items():
        if authority.authority_id != authority.expected_authority_id():
            raise ValueError(f"stale implementation authority identity: {kind}")
        closure = build_dependency_closure(authority.entrypoint, source_root)
        if authority.source_tree_sha256 != closure.source_tree_sha256:
            raise ValueError(f"stale implementation source authority: {kind}")
        if authority.dependency_closure_sha256 != closure.dependency_closure_sha256:
            raise ValueError(f"stale implementation dependency closure: {kind}")
        closures[kind] = closure
    if len({authority.entrypoint for authority in authorities}) != len(authorities):
        raise ValueError("implementation roles must use distinct entrypoints")
    if len({authority.source_tree_sha256 for authority in authorities}) != len(authorities):
        raise ValueError("implementation roles must have distinct source authorities")
    if len({authority.algorithm_description_digest for authority in authorities}) != len(
        authorities
    ):
        raise ValueError("implementation roles must declare distinct algorithms")

    entrypoint_modules = {kind: closure.entrypoint_module_id for kind, closure in closures.items()}
    for kind, closure in closures.items():
        forbidden = set(entrypoint_modules.values()) - {entrypoint_modules[kind]}
        imported_roles = forbidden & {item.module_id for item in closure.modules}
        if imported_roles:
            raise ValueError(
                f"{kind} implementation imports another role: {sorted(imported_roles)}"
            )

    roles_by_module: dict[str, set[str]] = defaultdict(set)
    records_by_module: dict[str, DependencyModuleRecord] = {}
    for kind, closure in closures.items():
        for record in closure.modules:
            roles_by_module[record.module_id].add(kind)
            records_by_module[record.module_id] = record
    shared_modules = {
        module_id: roles for module_id, roles in roles_by_module.items() if len(roles) > 1
    }
    if reviewed_shared_module_ids != tuple(sorted(set(reviewed_shared_module_ids))):
        raise ValueError("reviewed shared-module allowlist must be sorted and unique")
    if any(
        authority.allowed_shared_module_ids != reviewed_shared_module_ids
        for authority in authorities
    ):
        raise ValueError("implementation authority differs from reviewed shared-module allowlist")
    unreviewed_shared_modules = set(shared_modules) - set(reviewed_shared_module_ids)
    if unreviewed_shared_modules:
        raise ValueError(
            "shared dependencies are outside the reviewed allowlist: "
            f"{sorted(unreviewed_shared_modules)}"
        )
    root = source_root.resolve(strict=True)
    for module_id, roles in shared_modules.items():
        undeclared = [
            kind for kind in roles if module_id not in by_kind[kind].allowed_shared_module_ids
        ]
        if undeclared:
            raise ValueError(f"undeclared shared dependency {module_id}: {sorted(undeclared)}")
        path = root / records_by_module[module_id].relative_path
        if _has_shared_semantic_authority(module_id, path):
            raise ValueError(f"shared semantic dependency is forbidden: {module_id}")

    authority = {
        "schema_version": "v7.2.implementation-independence.v1",
        "authority_ids": sorted(item.authority_id for item in authorities),
        "closure_ids": sorted(item.closure_id for item in closures.values()),
        "shared_module_ids": sorted(shared_modules),
        "qualified": True,
    }
    return ImplementationIndependenceEvidence.model_validate_json(
        canonical_json(
            {
                "evidence_id": content_id("v72-implementation-independence", authority),
                **authority,
            }
        )
    )


__all__ = [
    "DependencyModuleRecord",
    "ImplementationDependencyClosure",
    "ImplementationIndependenceEvidence",
    "audit_implementation_independence",
    "build_dependency_closure",
    "build_implementation_authority",
    "reviewed_shared_module_allowlist",
]
