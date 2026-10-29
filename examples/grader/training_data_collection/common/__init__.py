"""Version-neutral training-data pipeline primitives."""

from .filesystem import (
    atomic_directory,
    file_sha256,
    iter_regular_files,
    make_tree_read_only,
    make_tree_writable,
    resolve_under,
    safe_relative_path,
    tree_manifest,
    tree_sha256,
)
from .identity import (
    canonical_json,
    content_digest,
    content_id,
    derive_seed,
    sha256_bytes,
)

__all__ = [
    "atomic_directory",
    "canonical_json",
    "content_digest",
    "content_id",
    "derive_seed",
    "file_sha256",
    "iter_regular_files",
    "make_tree_read_only",
    "make_tree_writable",
    "resolve_under",
    "safe_relative_path",
    "sha256_bytes",
    "tree_manifest",
    "tree_sha256",
]
