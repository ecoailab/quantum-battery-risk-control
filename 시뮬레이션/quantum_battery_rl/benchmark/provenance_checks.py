"""Figure-chain and publication-text checks for provenance audits."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from re import compile as compile_regex

from .artifact_models import HeldOutEvaluationReceipt
from .figure_models import FigureReceipt, FigureSidecar
from .paper_values_runner import PaperValuesReceipt
from .provenance_models import PlotSeriesMapping
from .smoke_fitters import source_hash


LEGACY_RESULT_PATTERN = compile_regex(
    r"(?<![\d.])(0\.950|0\.808|0\.638|0\.935|26\.6%|91\.2%)(?!\d)"
)
SHA256_PATTERN = compile_regex(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
FIGURE_IDS = (
    "uncertainty_coverage",
    "convergence_cost",
    "heldout_distributions",
    "mean_cvar_comparison",
)


class ProvenanceAuditError(ValueError):
    """Raised when publication content cannot resolve to canonical evidence."""


def verify_figures(
    project_root: Path,
    canonical_root: Path,
    figure_root: Path,
    paper_receipt: PaperValuesReceipt,
    receipt: FigureReceipt,
) -> tuple[PlotSeriesMapping, ...]:
    expected_outputs = {
        f"{figure_id}.{suffix}"
        for figure_id in FIGURE_IDS
        for suffix in ("json", "pdf", "png")
    }
    if set(receipt.output_sha256) != expected_outputs:
        raise ProvenanceAuditError("figure_receipt.json: incomplete figure set")
    expected_sources = _current_figure_sources(canonical_root, paper_receipt)
    implementation = source_hash(
        project_root,
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
    if (
        receipt.source_artifact_sha256s != expected_sources
        or receipt.figure_implementation_sha256 != implementation
    ):
        raise ProvenanceAuditError("figure_receipt.json: stale canonical link")
    mappings: list[PlotSeriesMapping] = []
    for name, expected_hash in receipt.output_sha256.items():
        payload = read_required(figure_root / name)
        digest = sha256(payload).hexdigest()
        if digest != expected_hash:
            raise ProvenanceAuditError(f"{name}: stale figure output")
        if Path(name).suffix != ".json":
            continue
        sidecar = FigureSidecar.model_validate_json(payload)
        if not set(sidecar.source_artifact_sha256s).issubset(expected_sources):
            raise ProvenanceAuditError(f"{name}: legacy figure input")
        mappings.extend(_series_mappings(name, digest, sidecar))
    return tuple(mappings)


def _series_mappings(
    name: str,
    sidecar_sha256: str,
    sidecar: FigureSidecar,
) -> list[PlotSeriesMapping]:
    return [
        PlotSeriesMapping(
            figure_id=sidecar.figure_id,
            sidecar_path=name,
            sidecar_sha256=sidecar_sha256,
            series_id=series.series_id,
            panel=series.panel,
            source_artifact_sha256s=sidecar.source_artifact_sha256s,
            point_count=len(series.y),
            data_sha256=sha256(series.model_dump_json().encode()).hexdigest(),
        )
        for series in sidecar.series
    ]


def _current_figure_sources(
    canonical_root: Path,
    paper_receipt: PaperValuesReceipt,
) -> tuple[str, str, str, str]:
    draws = read_required(canonical_root / "draws" / "test.json")
    pilot = read_required(canonical_root / "pilot" / "budget_amendment.json")
    raw_receipt = HeldOutEvaluationReceipt.model_validate_json(
        read_required(canonical_root / "raw_results_receipt.json")
    )
    return (
        sha256(draws).hexdigest(),
        sha256(pilot).hexdigest(),
        raw_receipt.raw_results_sha256,
        paper_receipt.paper_values_sha256,
    )


def allowed_publication_hashes(
    paper: PaperValuesReceipt,
    paper_payload: bytes,
    figures: FigureReceipt,
    figure_payload: bytes,
) -> set[str]:
    allowed: set[str] = {
        paper.manifest_sha256,
        paper.statistics_sha256,
        paper.claim_decision_sha256,
        paper.paper_values_sha256,
        sha256(paper_payload).hexdigest(),
        figures.figure_implementation_sha256,
        sha256(figure_payload).hexdigest(),
    }
    allowed.update(str(value) for value in paper.generated_sha256.values())
    allowed.update(figures.source_artifact_sha256s)
    allowed.update(str(value) for value in figures.output_sha256.values())
    return allowed


def scan_publication_paths(paths: tuple[Path, ...], allowed_hashes: set[str]) -> None:
    for path in paths:
        lines = read_required(path).decode("utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            legacy = LEGACY_RESULT_PATTERN.search(line)
            if legacy is not None:
                raise ProvenanceAuditError(
                    f"{path.name}:{line_number}: legacy literal {legacy.group()}"
                )
            stale = next(
                (
                    match.group()
                    for match in SHA256_PATTERN.finditer(line)
                    if match.group() not in allowed_hashes
                ),
                None,
            )
            if stale is not None:
                raise ProvenanceAuditError(
                    f"{path.name}:{line_number}: stale hash {stale}"
                )


def read_required(path: Path) -> bytes:
    if not path.is_file():
        raise ProvenanceAuditError(f"{path.name}: missing publication input")
    return path.read_bytes()
