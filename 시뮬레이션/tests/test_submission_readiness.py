from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Final, TypedDict, cast


WORKSPACE_ROOT: Final = Path(__file__).parents[2]
READINESS_PATH: Final = WORKSPACE_ROOT / "paper" / "submission_readiness.json"


class ArtifactEntry(TypedDict):
    path: str
    sha256: str


class Blocker(TypedDict):
    id: str
    detail: str
    resolution: str


class Readiness(TypedDict):
    selected_journal: str
    publication_route: str
    mandatory_author_charges: int
    archive_identifier: str | None
    gates: dict[str, bool]
    artifacts: dict[str, ArtifactEntry]
    blockers: list[Blocker]
    ready: bool


def _readiness() -> Readiness:
    return cast(
        Readiness,
        json.loads(READINESS_PATH.read_text(encoding="utf-8")),
    )


def test_submission_readiness_pins_every_declared_artifact() -> None:
    readiness = _readiness()
    for artifact in readiness["artifacts"].values():
        path = WORKSPACE_ROOT / artifact["path"]
        assert path.is_file(), artifact["path"]
        assert sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_readiness_is_exactly_the_conjunction_of_gates() -> None:
    readiness = _readiness()
    gates = readiness["gates"]
    blockers = readiness["blockers"]
    expected = all(gates.values()) and not blockers
    assert readiness["ready"] is expected
    assert readiness["ready"] is True
    assert blockers == []
    assert all(gates.values())


def test_zero_fee_jpa_route_remains_locked() -> None:
    readiness = _readiness()
    assert readiness["selected_journal"] == (
        "Journal of Physics A: Mathematical and Theoretical"
    )
    assert readiness["publication_route"] == "subscription-access"
    assert readiness["mandatory_author_charges"] == 0
    assert readiness["archive_identifier"] == (
        "https://github.com/ecoailab/quantum-battery-risk-control/"
        "releases/tag/v1.0.0"
    )
