"""Resume-safe end-to-end canonical smoke benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .artifact_models import FitArtifact
from .artifact_store import ArtifactStore
from .controller import TrainingDraws, ValidationDraws
from .manifest import load_manifest
from .smoke_artifacts import (
    StaleSmokeArtifactError,
    fit_or_resume,
    load_fairness,
    load_source_draws,
    persist_subset,
)
from .smoke_evaluation import evaluate_or_resume
from .smoke_fitters import SmokeFitterSpec, build_smoke_fitter_specs


@dataclass(frozen=True, slots=True)
class SmokeRunReport:
    fit_hashes: tuple[tuple[str, str], ...]
    raw_results_sha256: str
    reused_fit_count: int
    raw_results_reused: bool

    @property
    def fit_sha256s(self) -> dict[str, str]:
        return dict(self.fit_hashes)


def run_smoke(
    project_root: Path,
    manifest_path: Path,
    canonical_root: Path,
) -> SmokeRunReport:
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(manifest_path)
    registry, certificate = load_fairness(canonical_root)
    registered_ids = tuple(method.method_id for method in registry.methods)
    if (
        not certificate.certified
        or certificate.manifest_sha256 != manifest_sha256
        or tuple(certificate.method_ids) != registered_ids
    ):
        raise StaleSmokeArtifactError("fairness certificate does not match the smoke run")
    train_source = load_source_draws(canonical_root, "train", manifest_sha256)
    validation_source = load_source_draws(
        canonical_root,
        "validation",
        manifest_sha256,
    )
    smoke_root = canonical_root / "smoke"
    smoke_store = ArtifactStore(smoke_root)
    train_set, train_hash = persist_subset(
        smoke_store,
        smoke_root,
        train_source,
        manifest.optimization.smoke_scenarios,
    )
    validation_set, validation_hash = persist_subset(
        smoke_store,
        smoke_root,
        validation_source,
        manifest.optimization.smoke_scenarios,
    )
    training = TrainingDraws(train_set, train_hash)
    validation = ValidationDraws(validation_set, validation_hash)
    specs = build_smoke_fitter_specs(
        project_root,
        manifest,
        manifest_sha256,
        validation,
    )
    if tuple(spec.method_id for spec in specs) != registered_ids:
        raise StaleSmokeArtifactError("smoke fitter set differs from certified registry")
    seed = manifest.optimization.optimizer_seeds[0]
    fitted: list[tuple[SmokeFitterSpec, FitArtifact, str]] = []
    reused_fit_count = 0
    for spec in specs:
        artifact, artifact_hash, reused = fit_or_resume(
            smoke_store,
            smoke_root,
            spec,
            training,
            validation_hash,
            manifest,
            manifest_sha256,
            seed,
        )
        fitted.append((spec, artifact, artifact_hash))
        reused_fit_count += int(reused)
    raw, raw_hash, raw_reused = evaluate_or_resume(
        smoke_store,
        smoke_root,
        fitted,
        validation_set,
        validation_hash,
        manifest,
        manifest_sha256,
        registry.methods[0].simulator_sha256,
    )
    if len(raw.records) != len(specs) * len(validation_set.draws):
        raise StaleSmokeArtifactError("smoke evaluation cardinality is incomplete")
    return SmokeRunReport(
        fit_hashes=tuple((spec.method_id, fit_hash) for spec, _, fit_hash in fitted),
        raw_results_sha256=raw_hash,
        reused_fit_count=reused_fit_count,
        raw_results_reused=raw_reused,
    )
