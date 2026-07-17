"""Canonical numeric source and deterministic TeX fragment generation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifact_models import ClaimDecisionArtifact, PaperValuesArtifact
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .claim_decision_runner import ClaimDecisionReceipt
from .manifest import load_manifest
from .paper_values_content import derive_values
from .paper_values_tex import render_fragments
from .smoke_fitters import source_hash
from .statistics_models import StatisticsArtifact, StatisticsReceipt


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StalePaperValuesError(ValueError):
    """Raised when canonical inputs or generated outputs are stale."""


class PaperValuesReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["1.0"]
    artifact_type: Literal["paper_values_receipt"]
    manifest_sha256: Sha256
    statistics_sha256: Sha256
    claim_decision_sha256: Sha256
    paper_values_sha256: Sha256
    paper_values_implementation_sha256: Sha256
    generated_sha256: dict[str, Sha256]


@dataclass(frozen=True, slots=True)
class PaperValuesRequest:
    project_root: Path
    manifest_path: Path
    canonical_root: Path
    output_root: Path
    generated_root: Path


@dataclass(frozen=True, slots=True)
class PaperValuesReport:
    paper_values_sha256: str
    receipt_sha256: str
    value_count: int
    reused: bool


def run_paper_values(request: PaperValuesRequest) -> PaperValuesReport:
    """Build or exactly resume the canonical publication-number bundle."""
    manifest_hash = sha256(request.manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(request.manifest_path)
    statistics_receipt = StatisticsReceipt.model_validate_json(
        (request.canonical_root / "statistics_receipt.json").read_bytes(),
    )
    statistics = ArtifactStore(request.canonical_root).read(
        StatisticsArtifact,
        ArtifactReadRequest(
            Path("statistics.json"), statistics_receipt.statistics_sha256
        ),
    )
    claim_receipt = ClaimDecisionReceipt.model_validate_json(
        (request.canonical_root / "claim_decision_receipt.json").read_bytes(),
    )
    claim = ArtifactStore(request.canonical_root).read(
        ClaimDecisionArtifact,
        ArtifactReadRequest(
            Path("claim_decision.json"), claim_receipt.claim_decision_sha256
        ),
    )
    implementation_hash = source_hash(
        request.project_root,
        tuple(
            Path("quantum_battery_rl/benchmark") / name
            for name in (
                "paper_values_content.py",
                "paper_values_tex.py",
                "paper_values_runner.py",
            )
        ),
    )
    resumed = _resume(
        request, manifest_hash, statistics_receipt, claim_receipt, implementation_hash
    )
    if resumed is not None:
        return resumed
    values = derive_values(
        (
            manifest.statistics.bootstrap_samples,
            manifest.statistics.confidence_level,
            manifest.statistics.cvar_alpha,
        ),
        (statistics_receipt.statistics_sha256, statistics),
        (claim_receipt.claim_decision_sha256, claim),
    )
    artifact = PaperValuesArtifact(
        schema_version="1.0",
        artifact_type="paper_values",
        manifest_sha256=manifest_hash,
        summary_sha256=statistics_receipt.statistics_sha256,
        claim_decision_sha256=claim_receipt.claim_decision_sha256,
        values=values,
    )
    artifact_receipt = ArtifactStore(request.output_root).write(
        Path("paper_values.json"), artifact
    )
    generated = render_fragments(artifact)
    request.generated_root.mkdir(parents=True, exist_ok=True)
    generated_hashes: dict[str, str] = {}
    for name, text in generated:
        payload = text.encode("utf-8")
        _write_atomic(request.generated_root / name, payload)
        generated_hashes[name] = sha256(payload).hexdigest()
    receipt = PaperValuesReceipt(
        schema_version="1.0",
        artifact_type="paper_values_receipt",
        manifest_sha256=manifest_hash,
        statistics_sha256=statistics_receipt.statistics_sha256,
        claim_decision_sha256=claim_receipt.claim_decision_sha256,
        paper_values_sha256=artifact_receipt.sha256,
        paper_values_implementation_sha256=implementation_hash,
        generated_sha256=generated_hashes,
    )
    receipt_receipt = ArtifactStore(request.output_root).write(
        Path("paper_values_receipt.json"),
        receipt,
    )
    return PaperValuesReport(
        artifact_receipt.sha256, receipt_receipt.sha256, len(values), False
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        _ = temporary.write_bytes(payload)
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resume(
    request: PaperValuesRequest,
    manifest_hash: str,
    statistics_receipt: StatisticsReceipt,
    claim_receipt: ClaimDecisionReceipt,
    implementation_hash: str,
) -> PaperValuesReport | None:
    artifact_path = request.output_root / "paper_values.json"
    receipt_path = request.output_root / "paper_values_receipt.json"
    if not artifact_path.is_file() and not receipt_path.is_file():
        return None
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise StalePaperValuesError("paper values and receipt must coexist")
    payload = receipt_path.read_bytes()
    receipt = PaperValuesReceipt.model_validate_json(payload)
    artifact = ArtifactStore(request.output_root).read(
        PaperValuesArtifact,
        ArtifactReadRequest(Path("paper_values.json"), receipt.paper_values_sha256),
    )
    if (
        receipt.manifest_sha256 != manifest_hash
        or receipt.statistics_sha256 != statistics_receipt.statistics_sha256
        or receipt.claim_decision_sha256 != claim_receipt.claim_decision_sha256
        or receipt.paper_values_implementation_sha256 != implementation_hash
        or artifact.summary_sha256 != statistics_receipt.statistics_sha256
        or artifact.claim_decision_sha256 != claim_receipt.claim_decision_sha256
    ):
        raise StalePaperValuesError("stale paper values")
    for name, expected in receipt.generated_sha256.items():
        path = request.generated_root / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise StalePaperValuesError("stale generated fragment")
    return PaperValuesReport(
        receipt.paper_values_sha256,
        sha256(payload).hexdigest(),
        len(artifact.values),
        True,
    )
