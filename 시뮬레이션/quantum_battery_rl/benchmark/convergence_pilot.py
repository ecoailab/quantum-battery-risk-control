"""Validation-only staged convergence pilot and budget selection."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .artifact_store import ArtifactStore
from .budget_models import (
    BudgetAmendmentArtifact,
    BudgetDecision,
    PilotStageResult,
    PlateauCriterion,
    assess_budget,
)
from .controller import TrainingDraws, ValidationDraws
from .manifest import CanonicalManifest, load_manifest
from .pilot_artifacts import pilot_fit_relative_path, run_or_resume_stage
from .pilot_fitters import FINITE_DIFFERENCE_METHODS, PILOT_METHODS, PPO_METHOD
from .smoke_artifacts import (
    StaleSmokeArtifactError,
    load_fairness,
    read_existing,
)
from .smoke_fitters import source_hash
from .uncertainty import DrawSet


CRITERION = PlateauCriterion(
    absolute_tolerance=0.002,
    required_consecutive_transitions=2,
    extension_multiplier=2,
    maximum_extensions=6,
)


def run_convergence_pilot(
    project_root: Path,
    manifest_path: Path,
    canonical_root: Path,
) -> BudgetAmendmentArtifact:
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(manifest_path)
    _, certificate = load_fairness(canonical_root)
    if not certificate.certified or certificate.manifest_sha256 != manifest_sha256:
        raise StaleSmokeArtifactError("convergence pilot requires current certification")
    fairness_hash = sha256(
        (canonical_root / "fairness_certificate.json").read_bytes(),
    ).hexdigest()
    smoke_root = canonical_root / "smoke"
    smoke_store = ArtifactStore(smoke_root)
    training_set, training_hash = _load_smoke_draws(smoke_store, smoke_root, "train")
    validation_set, validation_hash = _load_smoke_draws(
        smoke_store,
        smoke_root,
        "validation",
    )
    if (
        training_set.source_manifest_sha256 != manifest_sha256
        or validation_set.source_manifest_sha256 != manifest_sha256
    ):
        raise StaleSmokeArtifactError("pilot draws do not match the manifest")
    implementation_hash = _pilot_implementation_hash(project_root)
    pilot_root = canonical_root / "pilot"
    pilot_store = ArtifactStore(pilot_root)
    existing, _ = read_existing(
        pilot_store,
        pilot_root,
        Path("budget_amendment.json"),
        BudgetAmendmentArtifact,
    )
    if existing is not None:
        _validate_amendment(
            existing,
            pilot_root,
            manifest_sha256,
            training_hash,
            validation_hash,
            fairness_hash,
            implementation_hash,
        )
        return existing
    training = TrainingDraws(training_set, training_hash)
    validation = ValidationDraws(validation_set, validation_hash)
    all_stages: list[PilotStageResult] = []
    decisions: list[BudgetDecision] = []
    for method_id in PILOT_METHODS:
        stages, decision = _run_method_stages(
            project_root,
            pilot_store,
            pilot_root,
            manifest,
            manifest_sha256,
            training,
            validation,
            method_id,
        )
        all_stages.extend(stages)
        decisions.append(decision)
    amendment = BudgetAmendmentArtifact(
        schema_version="1.0",
        artifact_type="budget_amendment",
        manifest_sha256=manifest_sha256,
        training_draws_sha256=training_hash,
        validation_draws_sha256=validation_hash,
        fairness_certificate_sha256=fairness_hash,
        pilot_implementation_sha256=implementation_hash,
        criterion=CRITERION,
        stages=tuple(all_stages),
        decisions=tuple(decisions),
        test_data_accessed=False,
    )
    _ = pilot_store.write(Path("budget_amendment.json"), amendment)
    return amendment


def _run_method_stages(
    project_root: Path,
    store: ArtifactStore,
    pilot_root: Path,
    manifest: CanonicalManifest,
    manifest_sha256: str,
    training: TrainingDraws,
    validation: ValidationDraws,
    method_id: str,
) -> tuple[tuple[PilotStageResult, ...], BudgetDecision]:
    budgets = list(_initial_budgets(manifest, method_id))
    stages = [
        run_or_resume_stage(
            project_root,
            store,
            pilot_root,
            manifest,
            manifest_sha256,
            training,
            validation,
            method_id,
            budget,
        )
        for budget in budgets
    ]
    decision = assess_budget(tuple(stages), CRITERION)
    extensions = 0
    while decision.status == "extension-required" and extensions < CRITERION.maximum_extensions:
        if decision.next_budget is None:
            raise StaleSmokeArtifactError("extension decision lacks a next budget")
        stages.append(
            run_or_resume_stage(
                project_root,
                store,
                pilot_root,
                manifest,
                manifest_sha256,
                training,
                validation,
                method_id,
                decision.next_budget,
            ),
        )
        decision = assess_budget(tuple(stages), CRITERION)
        extensions += 1
    return tuple(stages), decision


def _initial_budgets(manifest: CanonicalManifest, method_id: str) -> tuple[int, ...]:
    if method_id in FINITE_DIFFERENCE_METHODS:
        return manifest.optimization.grape_evaluation_stages
    if method_id == PPO_METHOD:
        return manifest.optimization.ppo_environment_step_stages
    raise ValueError(f"unsupported pilot method: {method_id}")


def _load_smoke_draws(
    store: ArtifactStore,
    root: Path,
    split: str,
) -> tuple[DrawSet, str]:
    draw_set, draw_hash = read_existing(
        store,
        root,
        Path("draws") / f"{split}.json",
        DrawSet,
    )
    if draw_set is None or draw_set.split != split:
        raise StaleSmokeArtifactError(f"missing smoke {split} draws")
    return draw_set, draw_hash


def _pilot_implementation_hash(project_root: Path) -> str:
    return source_hash(
        project_root,
        (
            Path("quantum_battery_rl/benchmark/budget_models.py"),
            Path("quantum_battery_rl/benchmark/pilot_fitters.py"),
            Path("quantum_battery_rl/benchmark/pilot_artifacts.py"),
            Path("quantum_battery_rl/benchmark/convergence_pilot.py"),
        ),
    )


def _validate_amendment(
    amendment: BudgetAmendmentArtifact,
    pilot_root: Path,
    manifest_hash: str,
    training_hash: str,
    validation_hash: str,
    fairness_hash: str,
    implementation_hash: str,
) -> None:
    if (
        amendment.manifest_sha256 != manifest_hash
        or amendment.training_draws_sha256 != training_hash
        or amendment.validation_draws_sha256 != validation_hash
        or amendment.fairness_certificate_sha256 != fairness_hash
        or amendment.pilot_implementation_sha256 != implementation_hash
        or amendment.criterion != CRITERION
        or amendment.test_data_accessed
    ):
        raise StaleSmokeArtifactError("stale budget amendment")
    for stage in amendment.stages:
        path = pilot_root / pilot_fit_relative_path(
            stage.method_id,
            stage.requested_budget,
        )
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != stage.fit_artifact_sha256:
            raise StaleSmokeArtifactError("budget amendment has a stale fit link")
