"""Hash-linked canonical seed-level statistical analysis runner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .artifact_models import HeldOutEvaluationReceipt, HeldOutResultsArtifact
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .full_fit_runner import FullFitRequest, run_full_fits
from .held_out_support import load_fit_references
from .manifest import load_manifest
from .smoke_artifacts import load_fairness, load_source_draws
from .smoke_fitters import source_hash
from .statistics_analysis import StatisticsInputs, analyze_statistics
from .statistics_models import StatisticsArtifact, StatisticsReceipt


BOOTSTRAP_SEED: Final = 4404


class StaleStatisticsInputError(ValueError):
    """Raised when canonical statistical inputs or outputs disagree."""


@dataclass(frozen=True, slots=True)
class StatisticsRequest:
    project_root: Path
    manifest_path: Path
    canonical_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class StatisticsReport:
    statistics_sha256: str
    receipt_sha256: str
    seed_statistic_count: int
    method_count: int
    comparison_count: int
    reused: bool


def run_statistics(request: StatisticsRequest) -> StatisticsReport:
    manifest_sha256 = sha256(request.manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(request.manifest_path)
    raw_receipt_path = request.canonical_root / "raw_results_receipt.json"
    raw_receipt_payload = raw_receipt_path.read_bytes()
    raw_receipt = HeldOutEvaluationReceipt.model_validate_json(raw_receipt_payload)
    raw = ArtifactStore(request.canonical_root).read(
        HeldOutResultsArtifact,
        ArtifactReadRequest(Path("raw_results.json"), raw_receipt.raw_results_sha256),
    )
    if raw.manifest_sha256 != manifest_sha256 or raw.evaluation_split != "test":
        raise StaleStatisticsInputError("statistics require current held-out evidence")
    implementation_hash = _statistics_implementation_hash(request.project_root)
    existing = _resume_existing(
        request,
        manifest_sha256,
        raw_receipt.raw_results_sha256,
        sha256(raw_receipt_payload).hexdigest(),
        implementation_hash,
        manifest.statistics.bootstrap_samples,
        manifest.statistics.confidence_level,
        manifest.statistics.cvar_alpha,
    )
    if existing is not None:
        return existing
    registry, certificate = load_fairness(request.canonical_root)
    if not certificate.certified:
        raise StaleStatisticsInputError("statistics require certified method fairness")
    method_ids = tuple(method.method_id for method in registry.methods)
    optimizer_seeds = manifest.optimization.optimizer_seeds
    test_draws = load_source_draws(
        request.canonical_root,
        "test",
        manifest_sha256,
    )
    fit_report = run_full_fits(
        FullFitRequest(
            request.project_root,
            request.manifest_path,
            request.canonical_root,
        ),
    )
    references = load_fit_references(request.canonical_root, fit_report.fit_hashes)
    values = _value_cube(raw, method_ids, optimizer_seeds, tuple(
        draw.draw_id for draw in test_draws.draws
    ))
    severities = np.asarray(
        tuple(draw.severity_fraction for draw in test_draws.draws),
        dtype=np.float64,
    )
    components = analyze_statistics(
        StatisticsInputs(
            method_ids,
            optimizer_seeds,
            severities,
            values,
            tuple(reference.artifact.controller.metadata.cost for reference in references),
            manifest.statistics.cvar_alpha,
            manifest.statistics.bootstrap_samples,
            manifest.statistics.confidence_level,
            BOOTSTRAP_SEED,
        ),
    )
    artifact = StatisticsArtifact(
        schema_version="1.0",
        artifact_type="statistics",
        manifest_sha256=manifest_sha256,
        raw_results_sha256=raw_receipt.raw_results_sha256,
        raw_results_receipt_sha256=sha256(raw_receipt_payload).hexdigest(),
        statistics_implementation_sha256=implementation_hash,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_samples=manifest.statistics.bootstrap_samples,
        confidence_level=manifest.statistics.confidence_level,
        cvar_alpha=manifest.statistics.cvar_alpha,
        seed_statistics=components.seed_statistics,
        methods=components.methods,
        comparisons=components.comparisons,
    )
    store = ArtifactStore(request.output_root)
    artifact_receipt = store.write(Path("statistics.json"), artifact)
    receipt = StatisticsReceipt(
        schema_version="1.0",
        artifact_type="statistics_receipt",
        manifest_sha256=manifest_sha256,
        statistics_sha256=artifact_receipt.sha256,
        statistics_implementation_sha256=implementation_hash,
    )
    receipt_receipt = store.write(Path("statistics_receipt.json"), receipt)
    return _report(artifact, artifact_receipt.sha256, receipt_receipt.sha256, False)


def _value_cube(
    raw: HeldOutResultsArtifact,
    method_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    draw_ids: tuple[str, ...],
) -> NDArray[np.float64]:
    lookup = {
        (record.method_id, record.optimizer_seed, record.draw_id): record.final_ergotropy
        for record in raw.records
    }
    expected_count = len(method_ids) * len(seeds) * len(draw_ids)
    if len(lookup) != expected_count:
        raise StaleStatisticsInputError("held-out identity cardinality is incomplete")
    values = np.empty((len(method_ids), len(seeds), len(draw_ids)), dtype=np.float64)
    for method_index, method_id in enumerate(method_ids):
        for seed_index, seed in enumerate(seeds):
            for draw_index, draw_id in enumerate(draw_ids):
                key = (method_id, seed, draw_id)
                if key not in lookup:
                    raise StaleStatisticsInputError("held-out paired identity is missing")
                values[method_index, seed_index, draw_index] = lookup[key]
    values.setflags(write=False)
    return values


def _resume_existing(
    request: StatisticsRequest,
    manifest_hash: str,
    raw_hash: str,
    raw_receipt_hash: str,
    implementation_hash: str,
    bootstrap_samples: int,
    confidence_level: float,
    cvar_alpha: float,
) -> StatisticsReport | None:
    statistics_path = request.output_root / "statistics.json"
    receipt_path = request.output_root / "statistics_receipt.json"
    if not statistics_path.is_file() and not receipt_path.is_file():
        return None
    if not statistics_path.is_file() or not receipt_path.is_file():
        raise StaleStatisticsInputError("statistics artifact and receipt must coexist")
    receipt_payload = receipt_path.read_bytes()
    receipt = StatisticsReceipt.model_validate_json(receipt_payload)
    artifact = ArtifactStore(request.output_root).read(
        StatisticsArtifact,
        ArtifactReadRequest(Path("statistics.json"), receipt.statistics_sha256),
    )
    if (
        artifact.manifest_sha256 != manifest_hash
        or artifact.raw_results_sha256 != raw_hash
        or artifact.raw_results_receipt_sha256 != raw_receipt_hash
        or artifact.statistics_implementation_sha256 != implementation_hash
        or artifact.bootstrap_seed != BOOTSTRAP_SEED
        or artifact.bootstrap_samples != bootstrap_samples
        or artifact.confidence_level != confidence_level
        or artifact.cvar_alpha != cvar_alpha
        or len(artifact.seed_statistics) != 70
        or len(artifact.methods) != 7
        or len(artifact.comparisons) != 21
    ):
        raise StaleStatisticsInputError("stale statistics artifact")
    return _report(
        artifact,
        receipt.statistics_sha256,
        sha256(receipt_payload).hexdigest(),
        True,
    )


def _report(
    artifact: StatisticsArtifact,
    artifact_hash: str,
    receipt_hash: str,
    reused: bool,
) -> StatisticsReport:
    return StatisticsReport(
        artifact_hash,
        receipt_hash,
        len(artifact.seed_statistics),
        len(artifact.methods),
        len(artifact.comparisons),
        reused,
    )


def _statistics_implementation_hash(project_root: Path) -> str:
    return source_hash(
        project_root,
        tuple(
            Path("quantum_battery_rl/benchmark") / name
            for name in (
                "statistics_models.py",
                "statistics_core.py",
                "statistics_intervals.py",
                "statistics_comparisons.py",
                "statistics_analysis.py",
                "statistics_runner.py",
            )
        ),
    )
