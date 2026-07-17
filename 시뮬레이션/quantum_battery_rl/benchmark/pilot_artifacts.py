"""Fit persistence and validation scoring for convergence-pilot stages."""

from __future__ import annotations

from pathlib import Path

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence

from .artifact_store import ArtifactStore
from .budget_models import BudgetKind, PilotFitArtifact, PilotStageResult
from .controller import TrainingDraws, ValidationDraws
from .manifest import CanonicalManifest
from .pilot_fitters import PPO_METHOD, build_pilot_fitter
from .smoke_artifacts import StaleSmokeArtifactError, read_existing


def run_or_resume_stage(
    project_root: Path,
    store: ArtifactStore,
    pilot_root: Path,
    manifest: CanonicalManifest,
    manifest_sha256: str,
    training: TrainingDraws,
    validation: ValidationDraws,
    method_id: str,
    budget: int,
) -> PilotStageResult:
    kind: BudgetKind = (
        "environment_steps" if method_id == PPO_METHOD else "objective_evaluations"
    )
    spec = build_pilot_fitter(
        project_root,
        manifest,
        manifest_sha256,
        validation,
        method_id,
        budget,
    )
    relative = pilot_fit_relative_path(method_id, budget)
    existing, existing_hash = read_existing(
        store,
        pilot_root,
        relative,
        PilotFitArtifact,
    )
    if existing is None:
        controller = spec.fitter.fit(training, manifest.optimization.optimizer_seeds[0])
        existing = PilotFitArtifact(
            schema_version="1.0",
            artifact_type="pilot_fit",
            manifest_sha256=manifest_sha256,
            training_draws_sha256=training.draw_set_sha256,
            validation_draws_sha256=validation.draw_set_sha256,
            implementation_sha256=spec.implementation_sha256,
            budget_kind=kind,
            requested_budget=budget,
            controller=controller,
        )
        existing_hash = store.write(relative, existing).sha256
    _validate_stage_fit(
        existing,
        manifest,
        manifest_sha256,
        training,
        validation,
        spec.implementation_sha256,
        method_id,
        kind,
        budget,
    )
    values = tuple(
        simulate_pulse_sequence(
            existing.controller.pulse(manifest.physics.n_steps),
            T1=draw.t1,
            T2=draw.t2,
            omega_q=manifest.physics.omega_q,
            dt=manifest.physics.dt,
            n_steps=manifest.physics.n_steps,
        )[1]
        for draw in validation.draw_set.draws
    )
    cost = existing.controller.metadata.cost
    return PilotStageResult(
        method_id=method_id,
        budget_kind=kind,
        requested_budget=budget,
        objective_evaluations=cost.objective_evaluations,
        environment_steps=cost.environment_steps,
        validation_mean=sum(values) / len(values),
        fit_artifact_sha256=existing_hash,
    )


def _validate_stage_fit(
    artifact: PilotFitArtifact,
    manifest: CanonicalManifest,
    manifest_sha256: str,
    training: TrainingDraws,
    validation: ValidationDraws,
    implementation_sha256: str,
    method_id: str,
    kind: BudgetKind,
    budget: int,
) -> None:
    controller = artifact.controller
    if (
        artifact.manifest_sha256 != manifest_sha256
        or artifact.training_draws_sha256 != training.draw_set_sha256
        or artifact.validation_draws_sha256 != validation.draw_set_sha256
        or artifact.implementation_sha256 != implementation_sha256
        or artifact.budget_kind != kind
        or artifact.requested_budget != budget
        or controller.metadata.method_id != method_id
        or controller.metadata.provenance.optimizer_seed
        != manifest.optimization.optimizer_seeds[0]
        or controller.n_steps != manifest.physics.n_steps
        or controller.max_action != manifest.physics.max_omega
    ):
        raise StaleSmokeArtifactError(f"stale pilot fit for {method_id} at {budget}")


def pilot_fit_relative_path(method_id: str, budget: int) -> Path:
    if method_id == PPO_METHOD:
        return Path("fits") / method_id / "nested-validation" / f"{budget}.json"
    return Path("fits") / method_id / f"{budget}.json"
