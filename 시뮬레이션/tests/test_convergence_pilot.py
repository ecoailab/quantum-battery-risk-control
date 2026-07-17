from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

import quantum_battery_rl.benchmark.pilot_fitters as pilot_fitters_module
from quantum_battery_rl.benchmark import run_convergence_pilot, run_smoke
from quantum_battery_rl.benchmark.artifact_store import ArtifactStore
from quantum_battery_rl.benchmark.budget_models import (
    BudgetKind,
    PilotStageResult,
    PlateauCriterion,
    assess_budget,
)
from quantum_battery_rl.benchmark.fairness import (
    audit_registry,
    build_default_registry,
    canonical_simulator_sha256,
)
from quantum_battery_rl.benchmark.manifest import (
    OptimizationConfig,
    PhysicsConfig,
    load_manifest,
)
from quantum_battery_rl.benchmark.controller import ValidationDraws
from quantum_battery_rl.benchmark.pilot_fitters import PPO_METHOD, build_pilot_fitter
from quantum_battery_rl.benchmark.ppo_control import PpoControlConfig
from quantum_battery_rl.benchmark.uncertainty import DrawSet


PROJECT_ROOT: Final = Path(__file__).parents[1]
SOURCE_DRAW_ROOT: Final = PROJECT_ROOT / "results" / "canonical" / "draws"
PILOT_METHODS: Final = {
    "nominal-lbfgsb-finite-difference",
    "saa-mean-lbfgsb-finite-difference",
    "cvar-0.1-lbfgsb-finite-difference",
    "ppo-time-only-domain-randomized",
}


def _stage(
    method_id: str,
    kind: BudgetKind,
    budget: int,
    score: float,
) -> PilotStageResult:
    return PilotStageResult(
        method_id=method_id,
        budget_kind=kind,
        requested_budget=budget,
        objective_evaluations=budget if kind == "objective_evaluations" else 1,
        environment_steps=budget,
        validation_mean=score,
        fit_artifact_sha256="a" * 64,
    )


def _prepare_canonical_root(tmp_path: Path) -> tuple[Path, Path]:
    source_manifest = load_manifest(PROJECT_ROOT / "canonical_manifest.json")
    manifest = source_manifest.model_copy(
        update={
            "physics": PhysicsConfig(
                t1=100.0,
                t2=80.0,
                omega_q=5.0,
                max_omega=0.25,
                n_steps=2,
                dt=0.1,
            ),
            "optimization": OptimizationConfig(
                optimizer_seeds=(42,),
                grape_evaluation_stages=(8, 12, 16),
                ppo_environment_step_stages=(2, 4, 6),
                smoke_scenarios=4,
            ),
        },
    )
    manifest_path = tmp_path / "canonical_manifest.json"
    _ = manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    canonical_root = tmp_path / "canonical"
    draw_root = canonical_root / "draws"
    draw_root.mkdir(parents=True)
    for split in ("train", "validation"):
        source = DrawSet.model_validate_json(
            (SOURCE_DRAW_ROOT / f"{split}.json").read_bytes(),
        )
        draw_set = source.model_copy(update={"source_manifest_sha256": manifest_hash})
        _ = (draw_root / f"{split}.json").write_text(
            draw_set.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    registry = build_default_registry(
        manifest,
        manifest_hash,
        canonical_simulator_sha256(PROJECT_ROOT),
    )
    store = ArtifactStore(canonical_root)
    _ = store.write(Path("method_registry.json"), registry)
    _ = store.write(Path("fairness_certificate.json"), audit_registry(registry, manifest))
    _ = run_smoke(PROJECT_ROOT, manifest_path, canonical_root)
    return manifest_path, canonical_root


def test_budget_selection_requires_two_plateau_transitions() -> None:
    criterion = PlateauCriterion(
        absolute_tolerance=0.002,
        required_consecutive_transitions=2,
        extension_multiplier=2,
        maximum_extensions=2,
    )
    plateau = (
        _stage("mean", "objective_evaluations", 500, 0.70),
        _stage("mean", "objective_evaluations", 1000, 0.701),
        _stage("mean", "objective_evaluations", 2000, 0.702),
    )
    changing = (
        _stage("ppo", "environment_steps", 50_000, 0.40),
        _stage("ppo", "environment_steps", 100_000, 0.43),
        _stage("ppo", "environment_steps", 200_000, 0.45),
    )

    selected = assess_budget(plateau, criterion)
    extension = assess_budget(changing, criterion)

    assert selected.status == "plateau"
    assert selected.selected_budget == 2000
    assert selected.next_budget is None
    assert extension.status == "extension-required"
    assert extension.selected_budget is None
    assert extension.next_budget == 400_000


def test_pilot_ppo_uses_fixed_base_stage_checkpoint_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, canonical_root = _prepare_canonical_root(tmp_path)
    manifest = load_manifest(manifest_path)
    validation_path = canonical_root / "smoke" / "draws" / "validation.json"
    validation_set = DrawSet.model_validate_json(validation_path.read_bytes())
    validation = ValidationDraws(
        validation_set,
        sha256(validation_path.read_bytes()).hexdigest(),
    )

    captured: list[PpoControlConfig] = []

    class CapturingPpoFitter:
        def __init__(
            self,
            config: PpoControlConfig,
            validation_draws: ValidationDraws,
        ) -> None:
            _ = validation_draws
            captured.append(config)

    monkeypatch.setattr(
        pilot_fitters_module,
        "PpoControlFitter",
        CapturingPpoFitter,
    )
    _ = build_pilot_fitter(
        PROJECT_ROOT,
        manifest,
        sha256(manifest_path.read_bytes()).hexdigest(),
        validation,
        PPO_METHOD,
        budget=6,
    )

    assert captured[0].checkpoint_interval_steps == 2


def test_pilot_uses_manifest_stages_validation_only_and_resumes(tmp_path: Path) -> None:
    manifest_path, canonical_root = _prepare_canonical_root(tmp_path)

    first = run_convergence_pilot(PROJECT_ROOT, manifest_path, canonical_root)
    second = run_convergence_pilot(PROJECT_ROOT, manifest_path, canonical_root)

    assert {decision.method_id for decision in first.decisions} == PILOT_METHODS
    assert all(not stage.method_id.startswith("test-") for stage in first.stages)
    assert not first.test_data_accessed
    assert second == first
    assert not (canonical_root / "draws" / "test.json").exists()
    by_method = {
        method_id: tuple(
            stage.requested_budget
            for stage in first.stages
            if stage.method_id == method_id
        )
        for method_id in PILOT_METHODS
    }
    assert by_method["nominal-lbfgsb-finite-difference"][:3] == (8, 12, 16)
    assert by_method["ppo-time-only-domain-randomized"][:3] == (2, 4, 6)
    ppo_scores = tuple(
        stage.validation_mean
        for stage in first.stages
        if stage.method_id == "ppo-time-only-domain-randomized"
    )
    assert ppo_scores == tuple(sorted(ppo_scores)), (
        "larger PPO budgets must retain the best earlier validation checkpoint"
    )
