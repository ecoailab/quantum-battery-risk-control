"""Deterministic audit of canonical publication values and figures."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .artifact_models import PaperValuesArtifact
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .claim_decision_runner import ClaimDecisionReceipt
from .figure_models import FigureReceipt
from .paper_values_runner import PaperValuesReceipt
from .paper_values_tex import render_fragments
from .provenance_checks import ProvenanceAuditError as ProvenanceAuditError
from .provenance_checks import (
    FIGURE_IDS,
    allowed_publication_hashes,
    read_required,
    scan_publication_paths,
    verify_figures,
)
from .provenance_models import (
    GeneratedFragmentMapping,
    PublicationProvenanceArtifact,
    PublicationValueMapping,
)
from .smoke_fitters import source_hash
from .statistics_models import StatisticsReceipt


@dataclass(frozen=True, slots=True)
class ProvenanceAuditRequest:
    project_root: Path
    manifest_path: Path
    canonical_root: Path
    generated_root: Path
    figure_root: Path
    output_path: Path
    publication_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ProvenanceAuditReport:
    provenance_sha256: str
    value_mapping_count: int
    plot_point_count: int
    reused: bool


def run_provenance_audit(request: ProvenanceAuditRequest) -> ProvenanceAuditReport:
    """Verify and map the complete canonical publication surface."""
    paper_receipt_path = request.canonical_root / "paper_values_receipt.json"
    paper_receipt_payload = read_required(paper_receipt_path)
    paper_receipt = PaperValuesReceipt.model_validate_json(paper_receipt_payload)
    paper_values = ArtifactStore(request.canonical_root).read(
        PaperValuesArtifact,
        ArtifactReadRequest(
            Path("paper_values.json"), paper_receipt.paper_values_sha256
        ),
    )
    _verify_paper_chain(request, paper_receipt, paper_values)
    fragments = _verify_fragments(request, paper_receipt, paper_values)
    figure_receipt_path = request.figure_root / "figure_receipt.json"
    figure_receipt_payload = read_required(figure_receipt_path)
    figure_receipt = FigureReceipt.model_validate_json(figure_receipt_payload)
    plot_series = verify_figures(
        request.project_root,
        request.canonical_root,
        request.figure_root,
        paper_receipt,
        figure_receipt,
    )
    allowed_hashes = allowed_publication_hashes(
        paper_receipt,
        paper_receipt_payload,
        figure_receipt,
        figure_receipt_payload,
    )
    scan_publication_paths(request.publication_paths, allowed_hashes)
    publication_sources = {
        path.resolve()
        .relative_to(request.project_root.parent.resolve())
        .as_posix(): sha256(read_required(path)).hexdigest()
        for path in sorted(request.publication_paths, key=lambda item: item.as_posix())
    }
    values = tuple(
        PublicationValueMapping(
            key=item.key,
            value=item.value,
            formatted=item.formatted,
            source_artifact_sha256=item.source_artifact_sha256,
            source_selector=item.source_selector,
        )
        for item in paper_values.values
    )
    artifact = PublicationProvenanceArtifact(
        schema_version="1.0",
        artifact_type="publication_provenance",
        paper_values_receipt_sha256=sha256(paper_receipt_payload).hexdigest(),
        paper_values_sha256=paper_receipt.paper_values_sha256,
        figure_receipt_sha256=sha256(figure_receipt_payload).hexdigest(),
        publication_source_sha256=publication_sources,
        value_mapping_count=len(values),
        fragment_count=len(fragments),
        figure_count=len(FIGURE_IDS),
        plot_point_count=sum(item.point_count for item in plot_series),
        value_mappings=values,
        fragments=fragments,
        plot_series=plot_series,
        unmapped_inputs=(),
        stale_inputs=(),
        legacy_inputs=(),
    )
    payload = (artifact.model_dump_json(indent=2) + "\n").encode()
    reused = _write_or_resume(request.output_path, payload)
    return ProvenanceAuditReport(
        sha256(payload).hexdigest(), len(values), artifact.plot_point_count, reused
    )


def _verify_paper_chain(
    request: ProvenanceAuditRequest,
    receipt: PaperValuesReceipt,
    artifact: PaperValuesArtifact,
) -> None:
    statistics = StatisticsReceipt.model_validate_json(
        read_required(request.canonical_root / "statistics_receipt.json")
    )
    claim = ClaimDecisionReceipt.model_validate_json(
        read_required(request.canonical_root / "claim_decision_receipt.json")
    )
    implementation = source_hash(
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
    current = (
        sha256(request.manifest_path.read_bytes()).hexdigest(),
        statistics.statistics_sha256,
        claim.claim_decision_sha256,
        implementation,
    )
    recorded = (
        receipt.manifest_sha256,
        receipt.statistics_sha256,
        receipt.claim_decision_sha256,
        receipt.paper_values_implementation_sha256,
    )
    if current != recorded:
        raise ProvenanceAuditError("paper_values_receipt.json: stale canonical link")
    allowed = set(current[:3])
    unknown = [
        item.key
        for item in artifact.values
        if item.source_artifact_sha256 not in allowed
    ]
    if unknown:
        raise ProvenanceAuditError(f"paper_values.json: unmapped key {unknown[0]}")


def _verify_fragments(
    request: ProvenanceAuditRequest,
    receipt: PaperValuesReceipt,
    artifact: PaperValuesArtifact,
) -> tuple[GeneratedFragmentMapping, ...]:
    expected = dict(render_fragments(artifact))
    if set(expected) != set(receipt.generated_sha256):
        raise ProvenanceAuditError("paper_values_receipt.json: incomplete fragment set")
    mappings: list[GeneratedFragmentMapping] = []
    for name, text in expected.items():
        path = request.generated_root / name
        payload = read_required(path)
        digest = sha256(payload).hexdigest()
        if payload != text.encode() or digest != receipt.generated_sha256[name]:
            raise ProvenanceAuditError(f"{name}: stale generated fragment")
        keys = _fragment_keys(name, artifact)
        mappings.append(
            GeneratedFragmentMapping(path=name, sha256=digest, source_value_keys=keys)
        )
    return tuple(mappings)


def _fragment_keys(name: str, artifact: PaperValuesArtifact) -> tuple[str, ...]:
    keys = tuple(item.key for item in artifact.values)
    if name == "results_macros.tex":
        return keys
    prefixes = {
        "method_summary_table.tex": ("method.",),
        "severity_table.tex": ("severity.",),
        "comparison_table.tex": ("comparison.",),
        "cost_table.tex": ("cost.",),
    }[name]
    if name == "comparison_table.tex":
        return tuple(
            key
            for key in keys
            if key.startswith(prefixes) and key.endswith(".mean_difference.estimate")
        )
    if name == "method_summary_table.tex":
        metrics = (".mean.estimate", ".cvar_0_1.estimate", ".minimum.estimate")
        return tuple(
            key for key in keys if key.startswith(prefixes) and key.endswith(metrics)
        )
    if name == "severity_table.tex":
        metrics = (".mean.estimate", ".cvar_0_1.estimate")
        return tuple(
            key for key in keys if key.startswith(prefixes) and key.endswith(metrics)
        )
    return tuple(key for key in keys if key.startswith(prefixes))


def _write_or_resume(path: Path, payload: bytes) -> bool:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ProvenanceAuditError(f"{path.name}: stale provenance artifact")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        _ = temporary.write_bytes(payload)
        _ = temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return False
