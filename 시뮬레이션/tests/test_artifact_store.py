from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_store import (
    ArtifactPathError,
    ArtifactReadRequest,
    ArtifactStore,
    CorruptArtifactError,
    InvalidExpectedHashError,
    MissingArtifactError,
    StaleArtifactError,
)
from quantum_battery_rl.benchmark.uncertainty import DrawSet


TRAIN_PATH: Final = Path(__file__).parents[1] / "results" / "canonical" / "draws" / "train.json"


def _train_draws() -> DrawSet:
    return DrawSet.model_validate_json(TRAIN_PATH.read_bytes())


def test_store_writes_atomically_and_reads_exact_hash(tmp_path: Path) -> None:
    # Given
    store = ArtifactStore(tmp_path / "canonical")
    relative_path = Path("draws/train.json")
    artifact = _train_draws()

    # When
    first = store.write(relative_path, artifact)
    first_bytes = (tmp_path / "canonical" / relative_path).read_bytes()
    second = store.write(relative_path, artifact)
    restored = store.read(
        DrawSet,
        ArtifactReadRequest(relative_path, second.sha256),
    )

    # Then
    assert first.sha256 == second.sha256 == sha256(first_bytes).hexdigest()
    assert first.size_bytes == len(first_bytes)
    assert restored == artifact
    assert not (tmp_path / "canonical" / "draws" / ".train.json.tmp").exists()


def test_store_rejects_stale_hash_before_parsing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "canonical")
    relative_path = Path("draws/train.json")
    _ = store.write(relative_path, _train_draws())

    with pytest.raises(StaleArtifactError):
        _ = store.read(
            DrawSet,
            ArtifactReadRequest(relative_path, "0" * 64),
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"{not-json",
        b'{"split":"train","draws":[]}',
    ],
)
def test_store_rejects_corrupt_partial_and_legacy_payloads(
    tmp_path: Path,
    payload: bytes,
) -> None:
    root = tmp_path / "canonical"
    target = root / "draws" / "train.json"
    target.parent.mkdir(parents=True)
    _ = target.write_bytes(payload)
    request = ArtifactReadRequest(
        Path("draws/train.json"),
        sha256(payload).hexdigest(),
    )

    with pytest.raises(CorruptArtifactError):
        _ = ArtifactStore(root).read(DrawSet, request)


def test_store_blocks_traversal_absolute_paths_and_bad_hashes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "canonical")
    artifact = _train_draws()

    with pytest.raises(ArtifactPathError):
        _ = store.write(Path("../outside.json"), artifact)
    with pytest.raises(ArtifactPathError):
        _ = store.write(tmp_path / "outside.json", artifact)
    with pytest.raises(InvalidExpectedHashError):
        _ = ArtifactReadRequest(Path("draws/train.json"), "bad-hash")


def test_missing_exact_path_never_falls_back_to_legacy(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    legacy = root / "legacy" / "train.json"
    legacy.parent.mkdir(parents=True)
    _ = legacy.write_bytes(TRAIN_PATH.read_bytes())
    store = ArtifactStore(root)

    with pytest.raises(MissingArtifactError):
        _ = store.read(
            DrawSet,
            ArtifactReadRequest(Path("draws/train.json"), "0" * 64),
        )
