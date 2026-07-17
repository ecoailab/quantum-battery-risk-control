from __future__ import annotations

from pathlib import Path
from typing import Final

from quantum_battery_rl.benchmark.controller import ControllerFitter, TrainingDraws
from quantum_battery_rl.benchmark.heuristic_control import (
    BangBangFitter,
    HeuristicControlConfig,
    RandomReferenceFitter,
    SinusoidalFitter,
)
from quantum_battery_rl.benchmark.manifest import PhysicsConfig
from quantum_battery_rl.benchmark.uncertainty import DrawSet


TRAIN_PATH: Final = Path(__file__).parents[1] / "results" / "canonical" / "draws" / "train.json"


def _training_draws() -> TrainingDraws:
    source = DrawSet.model_validate_json(TRAIN_PATH.read_bytes())
    draw_set = DrawSet(
        schema_version="1.0",
        split="train",
        source_manifest_sha256=source.source_manifest_sha256,
        draws=source.draws[:2],
    )
    return TrainingDraws(draw_set, "c" * 64)


def _config() -> HeuristicControlConfig:
    return HeuristicControlConfig(
        physics=PhysicsConfig(
            t1=100.0,
            t2=80.0,
            omega_q=5.0,
            max_omega=0.25,
            n_steps=3,
            dt=0.1,
        ),
        manifest_sha256="a" * 64,
        implementation_sha256="b" * 64,
        axis_count=2,
        frequency_count=3,
        phase_count=2,
    )


def test_tuned_heuristics_are_deterministic_complete_and_costed() -> None:
    training = _training_draws()
    cases: tuple[tuple[ControllerFitter, str, int], ...] = (
        (BangBangFitter(_config()), "bang-bang-ensemble-grid", 6),
        (SinusoidalFitter(_config()), "sinusoidal-ensemble-grid", 6),
    )

    for fitter, method_id, expected_evaluations in cases:
        first = fitter.fit(training, seed=42)
        second = fitter.fit(training, seed=42)
        assert first.controls == second.controls
        assert first.metadata.method_id == method_id
        assert first.metadata.cost.objective_evaluations == expected_evaluations
        assert first.metadata.cost.environment_steps == expected_evaluations * 3 * 2
        assert len(first.metadata.objective_contributions) == 2
        assert not first.pulse(3).flags.writeable


def test_random_reference_is_seeded_frozen_and_not_tuned() -> None:
    training = _training_draws()
    fitter = RandomReferenceFitter(_config())

    first = fitter.fit(training, seed=42)
    repeated = fitter.fit(training, seed=42)
    different = fitter.fit(training, seed=43)

    assert first.controls == repeated.controls
    assert first.controls != different.controls
    assert first.metadata.method_id == "random-seeded-reference"
    assert first.metadata.cost.objective_evaluations == 1
    assert first.metadata.cost.gradient_evaluations == 0
    assert len(first.metadata.objective_contributions) == 2
    assert not first.pulse(3).flags.writeable
