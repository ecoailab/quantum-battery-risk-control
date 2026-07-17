"""Exact-hash orchestration for canonical publication figures."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .artifact_models import (
    HeldOutEvaluationReceipt,
    HeldOutResultsArtifact,
    PaperValuesArtifact,
)
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .budget_models import BudgetAmendmentArtifact
from .figure_data import FigureInputs, build_sidecars
from .figure_models import FigureReceipt, FigureSidecar
from .figure_render import render_figure
from .paper_values_runner import PaperValuesReceipt
from .smoke_fitters import source_hash
from .uncertainty import DrawSet


class StaleFigureError(ValueError):
    """Raised when canonical figure inputs or outputs are stale."""


@dataclass(frozen=True, slots=True)
class FigureRequest:
    project_root: Path
    canonical_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class FigureReport:
    receipt_sha256: str
    output_sha256: dict[str, str]
    figure_count: int
    reused: bool


def run_figures(request: FigureRequest) -> FigureReport:
    """Build or exactly resume all declared canonical figures."""
    implementation_hash = source_hash(
        request.project_root,
        tuple(
            Path("quantum_battery_rl/benchmark") / name
            for name in (
                "figure_models.py",
                "figure_data.py",
                "figure_render.py",
                "figure_runner.py",
            )
        ),
    )
    inputs = _load_inputs(request.canonical_root)
    source_hashes = (
        inputs.draws_sha256,
        inputs.pilot_sha256,
        inputs.raw_sha256,
        inputs.paper_values_sha256,
    )
    existing = _resume(request, implementation_hash, source_hashes)
    if existing is not None:
        return existing
    request.output_root.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for sidecar in build_sidecars(inputs):
        sidecar_receipt = ArtifactStore(request.output_root).write(
            Path(f"{sidecar.figure_id}.json"),
            sidecar,
        )
        output_hashes[f"{sidecar.figure_id}.json"] = sidecar_receipt.sha256
        png, pdf = render_figure(sidecar, request.output_root)
        output_hashes[png.name] = sha256(png.read_bytes()).hexdigest()
        output_hashes[pdf.name] = sha256(pdf.read_bytes()).hexdigest()
    receipt = FigureReceipt(
        schema_version="1.0",
        artifact_type="figure_receipt",
        figure_implementation_sha256=implementation_hash,
        source_artifact_sha256s=source_hashes,
        output_sha256=output_hashes,
    )
    receipt_receipt = ArtifactStore(request.output_root).write(
        Path("figure_receipt.json"), receipt
    )
    return FigureReport(receipt_receipt.sha256, output_hashes, 4, False)


def _load_inputs(canonical_root: Path) -> FigureInputs:
    draws_path = canonical_root / "draws" / "test.json"
    draws_payload = draws_path.read_bytes()
    draws = DrawSet.model_validate_json(draws_payload)
    pilot_path = canonical_root / "pilot" / "budget_amendment.json"
    pilot_payload = pilot_path.read_bytes()
    pilot = BudgetAmendmentArtifact.model_validate_json(pilot_payload)
    raw_receipt = HeldOutEvaluationReceipt.model_validate_json(
        (canonical_root / "raw_results_receipt.json").read_bytes(),
    )
    raw = ArtifactStore(canonical_root).read(
        HeldOutResultsArtifact,
        ArtifactReadRequest(Path("raw_results.json"), raw_receipt.raw_results_sha256),
    )
    values_receipt = PaperValuesReceipt.model_validate_json(
        (canonical_root / "paper_values_receipt.json").read_bytes(),
    )
    values = ArtifactStore(canonical_root).read(
        PaperValuesArtifact,
        ArtifactReadRequest(
            Path("paper_values.json"), values_receipt.paper_values_sha256
        ),
    )
    return FigureInputs(
        sha256(draws_payload).hexdigest(),
        draws,
        sha256(pilot_payload).hexdigest(),
        pilot,
        raw_receipt.raw_results_sha256,
        raw,
        values_receipt.paper_values_sha256,
        values,
    )


def _resume(
    request: FigureRequest,
    implementation_hash: str,
    source_hashes: tuple[str, str, str, str],
) -> FigureReport | None:
    receipt_path = request.output_root / "figure_receipt.json"
    if not receipt_path.is_file():
        return None
    payload = receipt_path.read_bytes()
    receipt = FigureReceipt.model_validate_json(payload)
    if (
        receipt.figure_implementation_sha256 != implementation_hash
        or receipt.source_artifact_sha256s != source_hashes
    ):
        raise StaleFigureError("stale figure inputs or implementation")
    for name, expected in receipt.output_sha256.items():
        path = request.output_root / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise StaleFigureError("stale figure output")
        if path.suffix == ".json":
            _ = FigureSidecar.model_validate_json(path.read_bytes())
    return FigureReport(sha256(payload).hexdigest(), receipt.output_sha256, 4, True)
