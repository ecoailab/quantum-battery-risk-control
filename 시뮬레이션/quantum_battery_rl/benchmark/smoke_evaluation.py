"""Common validation evaluation for canonical smoke fits."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence

from .artifact_models import EvaluationRecord, FitArtifact, RawResultsArtifact
from .artifact_store import ArtifactStore
from .controller import FittedController
from .manifest import CanonicalManifest
from .smoke_artifacts import StaleSmokeArtifactError, read_existing
from .smoke_fitters import SmokeFitterSpec
from .uncertainty import DrawSet, UncertaintyDraw


def evaluate_or_resume(
    store: ArtifactStore,
    root: Path,
    fitted: list[tuple[SmokeFitterSpec, FitArtifact, str]],
    validation: DrawSet,
    validation_hash: str,
    manifest: CanonicalManifest,
    manifest_hash: str,
    simulator_hash: str,
) -> tuple[RawResultsArtifact, str, bool]:
    relative = Path("raw_results.json")
    existing, existing_hash = read_existing(store, root, relative, RawResultsArtifact)
    fit_hashes = tuple(fit_hash for _, _, fit_hash in fitted)
    expected_identities = {
        (spec.method_id, manifest.optimization.optimizer_seeds[0], draw.draw_id)
        for spec, _, _ in fitted
        for draw in validation.draws
    }
    if existing is not None:
        identities = {
            (record.method_id, record.optimizer_seed, record.draw_id)
            for record in existing.records
        }
        if (
            existing.manifest_sha256 != manifest_hash
            or existing.evaluation_split != "validation"
            or existing.evaluation_draws_sha256 != validation_hash
            or existing.fit_artifact_sha256s != fit_hashes
            or identities != expected_identities
            or not _records_match(existing, fitted, validation, simulator_hash)
        ):
            raise StaleSmokeArtifactError("stale smoke raw results")
        return existing, existing_hash, True
    records = tuple(
        _evaluate_record(spec, artifact.controller, fit_hash, draw, manifest, simulator_hash)
        for spec, artifact, fit_hash in fitted
        for draw in validation.draws
    )
    artifact = RawResultsArtifact(
        schema_version="1.0",
        artifact_type="raw_results",
        manifest_sha256=manifest_hash,
        evaluation_split="validation",
        evaluation_draws_sha256=validation_hash,
        fit_artifact_sha256s=fit_hashes,
        records=records,
    )
    receipt = store.write(relative, artifact)
    return artifact, receipt.sha256, False


def _evaluate_record(
    spec: SmokeFitterSpec,
    controller: FittedController,
    fit_hash: str,
    draw: UncertaintyDraw,
    manifest: CanonicalManifest,
    simulator_hash: str,
) -> EvaluationRecord:
    pulse = controller.pulse(manifest.physics.n_steps)
    started = perf_counter()
    ergotropy = simulate_pulse_sequence(
        pulse,
        T1=draw.t1,
        T2=draw.t2,
        omega_q=manifest.physics.omega_q,
        dt=manifest.physics.dt,
        n_steps=manifest.physics.n_steps,
    )[1]
    runtime = perf_counter() - started
    return EvaluationRecord(
        method_id=spec.method_id,
        optimizer_seed=controller.metadata.provenance.optimizer_seed,
        draw_id=draw.draw_id,
        severity_fraction=draw.severity_fraction,
        final_ergotropy=ergotropy,
        runtime_seconds=runtime,
        pulse_sha256=_pulse_sha256(controller),
        fit_artifact_sha256=fit_hash,
        simulator_sha256=simulator_hash,
    )


def _records_match(
    raw: RawResultsArtifact,
    fitted: list[tuple[SmokeFitterSpec, FitArtifact, str]],
    validation: DrawSet,
    simulator_hash: str,
) -> bool:
    fit_contracts = {
        spec.method_id: (fit_hash, _pulse_sha256(artifact.controller))
        for spec, artifact, fit_hash in fitted
    }
    severities = {draw.draw_id: draw.severity_fraction for draw in validation.draws}
    return all(
        record.method_id in fit_contracts
        and record.draw_id in severities
        and record.fit_artifact_sha256 == fit_contracts[record.method_id][0]
        and record.pulse_sha256 == fit_contracts[record.method_id][1]
        and record.severity_fraction == severities[record.draw_id]
        and record.simulator_sha256 == simulator_hash
        for record in raw.records
    )


def _pulse_sha256(controller: FittedController) -> str:
    payload = json.dumps(controller.controls, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()
