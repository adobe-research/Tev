"""Safe, deterministic filesystem primitives for training-data packages."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from .identity import content_digest


def safe_relative_path(value: str) -> PurePosixPath:
    """Validate one nonempty package-relative POSIX path."""
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"unsafe package path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("~")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    return path


def resolve_under(root: Path, relative: str) -> Path:
    """Resolve a validated relative path beneath an existing authority root."""
    root = root.resolve(strict=True)
    candidate = root.joinpath(*safe_relative_path(relative).parts).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"package path escapes authority root: {relative!r}")
    return candidate


def file_sha256(path: Path) -> str:
    """Hash one regular, non-symlink file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"tree authority requires a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_regular_files(root: Path) -> tuple[Path, ...]:
    """Return every regular file in stable path order and reject symlinks."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"tree authority root must be a directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in authority trees: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"unsupported filesystem entry in authority tree: {path}")
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def tree_manifest(root: Path) -> tuple[dict[str, int | str], ...]:
    """Describe a tree using only content-bound, host-independent fields."""
    return tuple(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in iter_regular_files(root)
    )


def tree_sha256(root: Path) -> str:
    """Return the canonical content digest of a symlink-free tree."""
    return content_digest(list(tree_manifest(root)))


def prepare_empty_output_root(root: Path) -> None:
    """Create or validate the exact empty directory reserved for solver outputs."""
    unresolved = Path(os.path.abspath(root))
    try:
        resolved = root.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise FileExistsError(
            f"output root must be absent or an existing nonsymlink empty directory: {root}"
        ) from error
    if root.is_symlink() or unresolved != resolved:
        raise FileExistsError(
            f"output root must be absent or an existing nonsymlink empty directory: {root}"
        )
    root = unresolved
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return
    if not root.is_dir() or any(root.iterdir()):
        raise FileExistsError(
            f"output root must be absent or an existing nonsymlink empty directory: {root}"
        )


def make_tree_read_only(root: Path) -> None:
    """Remove write bits from a validated tree."""
    iter_regular_files(root)
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def make_tree_writable(root: Path) -> None:
    """Restore owner write bits so a temporary authority tree can be cleaned up."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"tree authority root must be a directory: {root}")
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in authority trees: {path}")
        path.chmod(0o755 if path.is_dir() else 0o644)


@contextmanager
def atomic_directory(destination: Path, *, prefix: str) -> Iterator[Path]:
    """Yield a staging directory and atomically publish it after successful construction."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=destination.parent) as raw:
        stage = Path(raw) / "stage"
        stage.mkdir()
        yield stage
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(stage, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
