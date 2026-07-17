"""Fit-reference loading and resume validation for held-out evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .artifact_models import (
    HeldOutEvaluationReceipt,
    HeldOutResultsArtifact,
)
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .full_fit_models import FullFitArtifact
from .smoke_artifacts import read_existing
from .smoke_fitters import source_hash
from .uncertainty import DrawSet


class StaleHeldOutEvaluationError(ValueError):
    """Raised when persisted held-out evidence disagrees with current inputs."""


@dataclass(frozen=True, slots=True)
class FitReference:
    method_id: str
    optimizer_seed: int
    artifact: FullFitArtifact
    artifact_sha256: str
    pulse_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationLinks:
    manifest_sha256: str
    test_draws_sha256: str
    fairness_sha256: str
    full_fit_config_sha256: str
    evaluation_implementation_sha256: str
    simulator_sha256: str


@dataclass(frozen=True, slots=True)
class ExistingEvaluation:
    raw_results_sha256: str
    receipt_sha256: str
    record_count: int


def load_fit_references(
    canonical_root: Path,
    fit_hashes: tuple[tuple[str, int, str], ...],
) -> tuple[FitReference, ...]:
    fits_root = canonical_root / "fits"
    store = ArtifactStore(fits_root)
    return tuple(
        _fit_reference(store, fits_root, method_id, seed, artifact_hash)
        for method_id, seed, artifact_hash in fit_hashes
    )


def _fit_reference(
    store: ArtifactStore,
    root: Path,
    method_id: str,
    seed: int,
    artifact_hash: str,
) -> FitReference:
    relative = Path(method_id) / f"{seed}.json"
    artifact, loaded_hash = read_existing(
        store,
        root,
        relative,
        FullFitArtifact,
    )
    if artifact is None or loaded_hash != artifact_hash:
        raise StaleHeldOutEvaluationError(f"missing frozen fit {method_id}/{seed}")
    payload = json.dumps(artifact.controller.controls, separators=(",", ":")).encode()
    return FitReference(
        method_id,
        seed,
        artifact,
        artifact_hash,
        sha256(payload).hexdigest(),
    )


def resume_existing(
    root: Path,
    references: tuple[FitReference, ...],
    test_draws: DrawSet,
    links: EvaluationLinks,
) -> ExistingEvaluation | None:
    raw_path = root / "raw_results.json"
    receipt_path = root / "raw_results_receipt.json"
    if not raw_path.is_file() and not receipt_path.is_file():
        return None
    if not raw_path.is_file() or not receipt_path.is_file():
        raise StaleHeldOutEvaluationError("held-out result and receipt must coexist")
    receipt_payload = receipt_path.read_bytes()
    receipt = HeldOutEvaluationReceipt.model_validate_json(receipt_payload)
    raw = ArtifactStore(root).read(
        HeldOutResultsArtifact,
        ArtifactReadRequest(Path("raw_results.json"), receipt.raw_results_sha256),
    )
    _validate_existing(raw, references, test_draws, links)
    return ExistingEvaluation(
        receipt.raw_results_sha256,
        sha256(receipt_payload).hexdigest(),
        len(raw.records),
    )


def _validate_existing(
    raw: HeldOutResultsArtifact,
    references: tuple[FitReference, ...],
    test_draws: DrawSet,
    links: EvaluationLinks,
) -> None:
    fit_contracts = {
        (reference.method_id, reference.optimizer_seed): (
            reference.artifact_sha256,
            reference.pulse_sha256,
        )
        for reference in references
    }
    severities = {draw.draw_id: draw.severity_fraction for draw in test_draws.draws}
    expected_identities = {
        (method_id, seed, draw_id)
        for method_id, seed in fit_contracts
        for draw_id in severities
    }
    identities = {
        (record.method_id, record.optimizer_seed, record.draw_id)
        for record in raw.records
    }
    valid_records = all(
        (record.method_id, record.optimizer_seed) in fit_contracts
        and record.draw_id in severities
        and record.fit_artifact_sha256
        == fit_contracts[(record.method_id, record.optimizer_seed)][0]
        and record.pulse_sha256
        == fit_contracts[(record.method_id, record.optimizer_seed)][1]
        and record.severity_fraction == severities[record.draw_id]
        and record.simulator_sha256 == links.simulator_sha256
        for record in raw.records
    )
    if (
        raw.manifest_sha256 != links.manifest_sha256
        or raw.evaluation_draws_sha256 != links.test_draws_sha256
        or raw.fairness_certificate_sha256 != links.fairness_sha256
        or raw.full_fit_config_sha256 != links.full_fit_config_sha256
        or raw.evaluation_implementation_sha256
        != links.evaluation_implementation_sha256
        or raw.simulator_sha256 != links.simulator_sha256
        or raw.fit_artifact_sha256s
        != tuple(reference.artifact_sha256 for reference in references)
        or identities != expected_identities
        or not valid_records
    ):
        raise StaleHeldOutEvaluationError("stale held-out raw results")


def evaluation_implementation_hash(project_root: Path) -> str:
    return source_hash(
        project_root,
        (
            Path("quantum_battery_rl/benchmark/held_out_evaluation.py"),
            Path("quantum_battery_rl/benchmark/held_out_support.py"),
            Path("quantum_battery_rl/benchmark/batch_dynamics.py"),
            Path("quantum_battery_rl/env/quantum_dynamics.py"),
        ),
    )
