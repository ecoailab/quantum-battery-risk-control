"""Resume-safe held-out evaluation of all canonical frozen fits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from .artifact_models import (
    EvaluationRecord,
    HeldOutEvaluationReceipt,
    HeldOutResultsArtifact,
)
from .artifact_store import ArtifactStore
from .batch_dynamics import simulate_ensemble_pulse
from .full_fit_runner import FullFitRequest, run_full_fits
from .held_out_support import (
    EvaluationLinks,
    FitReference,
    StaleHeldOutEvaluationError,
    evaluation_implementation_hash,
    load_fit_references,
    resume_existing,
)
from .manifest import PhysicsConfig, load_manifest
from .smoke_artifacts import load_fairness, load_source_draws
from .uncertainty import DrawSet


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationRequest:
    project_root: Path
    manifest_path: Path
    canonical_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationReport:
    raw_results_sha256: str
    receipt_sha256: str
    record_count: int
    reused: bool


def run_held_out_evaluation(
    request: HeldOutEvaluationRequest,
) -> HeldOutEvaluationReport:
    manifest_sha256 = sha256(request.manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(request.manifest_path)
    fit_report = run_full_fits(
        FullFitRequest(
            request.project_root,
            request.manifest_path,
            request.canonical_root,
        ),
    )
    if len(fit_report.fit_hashes) != 70 or fit_report.reused_fit_count != 70:
        raise StaleHeldOutEvaluationError(
            "held-out evaluation requires 70 complete frozen fits",
        )
    registry, certificate = load_fairness(request.canonical_root)
    simulator_hashes = {method.simulator_sha256 for method in registry.methods}
    if len(simulator_hashes) != 1 or not certificate.certified:
        raise StaleHeldOutEvaluationError("held-out evaluation requires one certified simulator")
    test_draws = load_source_draws(
        request.canonical_root,
        "test",
        manifest_sha256,
    )
    links = EvaluationLinks(
        manifest_sha256,
        _file_hash(request.canonical_root / "draws" / "test.json"),
        _file_hash(request.canonical_root / "fairness_certificate.json"),
        _file_hash(request.canonical_root / "full_fit_config.json"),
        evaluation_implementation_hash(request.project_root),
        simulator_hashes.pop(),
    )
    references = load_fit_references(request.canonical_root, fit_report.fit_hashes)
    existing = resume_existing(request.output_root, references, test_draws, links)
    if existing is not None:
        return HeldOutEvaluationReport(
            existing.raw_results_sha256,
            existing.receipt_sha256,
            existing.record_count,
            True,
        )
    records = tuple(
        record
        for reference in references
        for record in _evaluate_fit(reference, test_draws, manifest.physics, links)
    )
    expected_count = len(references) * len(test_draws.draws)
    if len(records) != expected_count:
        raise StaleHeldOutEvaluationError("held-out evaluation cardinality is incomplete")
    raw = HeldOutResultsArtifact(
        schema_version="1.0",
        artifact_type="held_out_raw_results",
        manifest_sha256=links.manifest_sha256,
        evaluation_split="test",
        evaluation_draws_sha256=links.test_draws_sha256,
        fairness_certificate_sha256=links.fairness_sha256,
        full_fit_config_sha256=links.full_fit_config_sha256,
        evaluation_implementation_sha256=links.evaluation_implementation_sha256,
        simulator_sha256=links.simulator_sha256,
        fit_artifact_sha256s=tuple(
            reference.artifact_sha256 for reference in references
        ),
        records=records,
    )
    store = ArtifactStore(request.output_root)
    raw_receipt = store.write(Path("raw_results.json"), raw)
    receipt = HeldOutEvaluationReceipt(
        schema_version="1.0",
        artifact_type="held_out_evaluation_receipt",
        manifest_sha256=links.manifest_sha256,
        raw_results_sha256=raw_receipt.sha256,
        evaluation_implementation_sha256=links.evaluation_implementation_sha256,
    )
    receipt_receipt = store.write(Path("raw_results_receipt.json"), receipt)
    return HeldOutEvaluationReport(
        raw_receipt.sha256,
        receipt_receipt.sha256,
        len(records),
        False,
    )


def _evaluate_fit(
    reference: FitReference,
    test_draws: DrawSet,
    physics: PhysicsConfig,
    links: EvaluationLinks,
) -> tuple[EvaluationRecord, ...]:
    pulse = reference.artifact.controller.pulse(physics.n_steps)
    started = perf_counter()
    values = simulate_ensemble_pulse(pulse, test_draws.draws, physics).normalized_ergotropy
    per_draw_runtime = (perf_counter() - started) / len(test_draws.draws)
    return tuple(
        EvaluationRecord(
            method_id=reference.method_id,
            optimizer_seed=reference.optimizer_seed,
            draw_id=draw.draw_id,
            severity_fraction=draw.severity_fraction,
            final_ergotropy=value,
            runtime_seconds=per_draw_runtime,
            pulse_sha256=reference.pulse_sha256,
            fit_artifact_sha256=reference.artifact_sha256,
            simulator_sha256=links.simulator_sha256,
        )
        for draw, value in zip(test_draws.draws, values, strict=True)
    )


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
