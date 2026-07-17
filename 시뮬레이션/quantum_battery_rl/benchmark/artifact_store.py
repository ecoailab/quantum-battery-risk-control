"""Atomic exact-path storage for canonical benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from re import fullmatch
from typing import TypeVar

from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactPathError(ValueError):
    """Raised when an artifact path escapes the canonical root."""

    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Artifact path must remain under the canonical root: {path}")


class InvalidExpectedHashError(ValueError):
    """Raised when a read request does not contain a SHA-256 hash."""

    value: str

    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(f"Expected a lowercase SHA-256 hash; received {value!r}")


class MissingArtifactError(FileNotFoundError):
    """Raised when the exact requested canonical path is absent."""

    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Canonical artifact does not exist: {path}")


class StaleArtifactError(ValueError):
    """Raised when artifact bytes do not match the expected hash."""

    expected: str
    actual: str

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Artifact hash mismatch: expected {expected}, received {actual}")


class CorruptArtifactError(ValueError):
    """Raised when artifact bytes do not validate against their schema."""

    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Canonical artifact failed schema validation: {path}")


@dataclass(frozen=True, slots=True)
class ArtifactReadRequest:
    relative_path: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if fullmatch(r"[0-9a-f]{64}", self.expected_sha256) is None:
            raise InvalidExpectedHashError(self.expected_sha256)


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    relative_path: Path
    sha256: str
    size_bytes: int


class ArtifactStore:
    _root: Path

    def __init__(self, root: Path) -> None:
        self._root = root

    def write(self, relative_path: Path, artifact: BaseModel) -> ArtifactReceipt:
        destination = self._resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (artifact.model_dump_json(indent=2) + "\n").encode("utf-8")
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            _ = temporary.write_bytes(payload)
            _ = temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return ArtifactReceipt(
            relative_path=relative_path,
            sha256=sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def read(self, model_type: type[ModelT], request: ArtifactReadRequest) -> ModelT:
        source = self._resolve(request.relative_path)
        if not source.is_file():
            raise MissingArtifactError(source)
        payload = source.read_bytes()
        actual_hash = sha256(payload).hexdigest()
        if actual_hash != request.expected_sha256:
            raise StaleArtifactError(request.expected_sha256, actual_hash)
        try:
            return model_type.model_validate_json(payload)
        except ValidationError as error:
            raise CorruptArtifactError(source) from error

    def _resolve(self, relative_path: Path) -> Path:
        if relative_path.is_absolute() or relative_path.suffix != ".json":
            raise ArtifactPathError(relative_path)
        root = self._root.resolve()
        candidate = (root / relative_path).resolve()
        try:
            _ = candidate.relative_to(root)
        except ValueError as error:
            raise ArtifactPathError(relative_path) from error
        return candidate
