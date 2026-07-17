from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence
from quantum_battery_rl.benchmark.controller import ControllerFitter, TrainingDraws
from quantum_battery_rl.benchmark.manifest import PhysicsConfig
from quantum_battery_rl.benchmark.mean_control import MeanControlConfig, MeanControlFitter
from quantum_battery_rl.benchmark.uncertainty import DrawSet


TRAIN_PATH: Final = Path(__file__).parents[1] / "results" / "canonical" / "draws" / "train.json"


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        t1=100.0,
        t2=80.0,
        omega_q=5.0,
        max_omega=0.25,
        n_steps=2,
        dt=0.1,
    )


def _training_draws() -> TrainingDraws:
    source = DrawSet.model_validate_json(TRAIN_PATH.read_bytes())
    draw_set = DrawSet(
        schema_version="1.0",
        split="train",
        source_manifest_sha256=source.source_manifest_sha256,
        draws=source.draws[:2],
    )
    return TrainingDraws(draw_set, "d" * 64)


def _config() -> MeanControlConfig:
    return MeanControlConfig(
        physics=_physics(),
        manifest_sha256="a" * 64,
        implementation_sha256="b" * 64,
        max_iterations=30,
        max_objective_evaluations=300,
    )


def _ergotropy(pulse: np.ndarray, t1: float, t2: float) -> float:
    physics = _physics()
    return simulate_pulse_sequence(
        pulse,
        T1=t1,
        T2=t2,
        omega_q=physics.omega_q,
        dt=physics.dt,
        n_steps=physics.n_steps,
    )[1]


def test_mean_fitter_records_every_draw_and_exact_provenance() -> None:
    # Given
    training = _training_draws()
    fitter: ControllerFitter = MeanControlFitter(_config())

    # When
    fitted = fitter.fit(training, seed=42)

    # Then
    assert fitted.metadata.method_id == "saa-mean-lbfgsb-finite-difference"
    assert fitted.metadata.converged
    assert fitted.metadata.provenance.training_draws_sha256 == "d" * 64
    assert tuple(item.draw_id for item in fitted.metadata.objective_contributions) == tuple(
        draw.draw_id for draw in training.draw_set.draws
    )
    pulse = fitted.pulse(2)
    for item, draw in zip(
        fitted.metadata.objective_contributions,
        training.draw_set.draws,
        strict=True,
    ):
        np.testing.assert_allclose(item.value, _ergotropy(pulse, draw.t1, draw.t2))


def test_mean_fitter_is_deterministic_and_optimizes_arithmetic_mean() -> None:
    # Given
    training = _training_draws()
    fitter = MeanControlFitter(_config())
    zero = np.zeros((2, 2), dtype=np.float64)

    # When
    first = fitter.fit(training, seed=7)
    second = fitter.fit(training, seed=7)
    fitted_mean = sum(
        item.value for item in first.metadata.objective_contributions
    ) / len(first.metadata.objective_contributions)
    zero_mean = sum(
        _ergotropy(zero, draw.t1, draw.t2) for draw in training.draw_set.draws
    ) / len(training.draw_set.draws)

    # Then
    assert first.controls == second.controls
    assert fitted_mean >= zero_mean
    assert first.metadata.cost.environment_steps == (
        first.metadata.cost.objective_evaluations * 2 * len(training.draw_set.draws)
    )
