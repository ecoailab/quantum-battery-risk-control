from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence
from quantum_battery_rl.benchmark.controller import (
    ControllerFitter,
    PulseController,
    TrainingDraws,
)
from quantum_battery_rl.benchmark.manifest import PhysicsConfig
from quantum_battery_rl.benchmark.nominal_control import (
    NominalControlConfig,
    NominalControlFitter,
)
from quantum_battery_rl.benchmark.uncertainty import DrawSet


TRAIN_PATH: Final = Path(__file__).parents[1] / "results" / "canonical" / "draws" / "train.json"


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        t1=100.0,
        t2=80.0,
        omega_q=5.0,
        max_omega=0.25,
        n_steps=4,
        dt=0.1,
    )


def _config() -> NominalControlConfig:
    return NominalControlConfig(
        physics=_physics(),
        manifest_sha256="a" * 64,
        implementation_sha256="b" * 64,
        max_iterations=30,
        max_objective_evaluations=500,
    )


def _training_draws() -> TrainingDraws:
    draw_set = DrawSet.model_validate_json(TRAIN_PATH.read_bytes())
    return TrainingDraws(draw_set, "c" * 64)


def test_nominal_fitter_implements_common_protocol_and_provenance() -> None:
    # Given
    fitter: ControllerFitter = NominalControlFitter(_config())

    # When
    fitted = fitter.fit(_training_draws(), seed=42)
    pulse_controller: PulseController = fitted
    pulse = pulse_controller.pulse(4)

    # Then
    assert fitted.metadata.method_id == "nominal-lbfgsb-finite-difference"
    assert "upper" not in fitted.metadata.method_id
    assert fitted.metadata.converged
    assert fitted.metadata.provenance.optimizer_seed == 42
    assert fitted.metadata.provenance.training_draws_sha256 == "c" * 64
    assert fitted.metadata.cost.objective_evaluations > 0
    assert fitted.metadata.cost.environment_steps == (
        fitted.metadata.cost.objective_evaluations * 4
    )
    assert pulse.shape == (4, 2)
    assert not pulse.flags.writeable


def test_nominal_fitter_is_deterministic_and_ignores_draw_parameters() -> None:
    # Given
    training = _training_draws()
    first_draw = training.draw_set.draws[0].model_copy(
        update={"t1": 50.0, "t2": 25.0},
    )
    altered_set = DrawSet(
        schema_version="1.0",
        split="train",
        source_manifest_sha256=training.draw_set.source_manifest_sha256,
        draws=(first_draw,),
    )
    altered = TrainingDraws(altered_set, "d" * 64)
    fitter = NominalControlFitter(_config())

    # When
    first = fitter.fit(training, seed=7)
    second = fitter.fit(altered, seed=7)

    # Then
    assert first.controls == second.controls
    assert first.metadata.provenance.training_draws_sha256 != (
        second.metadata.provenance.training_draws_sha256
    )


def test_nominal_fit_is_not_worse_than_zero_pulse() -> None:
    # Given
    physics = _physics()
    fitted = NominalControlFitter(_config()).fit(_training_draws(), seed=11)
    zero = np.zeros((physics.n_steps, 2), dtype=np.float64)

    # When
    _, fitted_ergotropy = simulate_pulse_sequence(
        fitted.pulse(physics.n_steps),
        T1=physics.t1,
        T2=physics.t2,
        omega_q=physics.omega_q,
        dt=physics.dt,
        n_steps=physics.n_steps,
    )
    _, zero_ergotropy = simulate_pulse_sequence(
        zero,
        T1=physics.t1,
        T2=physics.t2,
        omega_q=physics.omega_q,
        dt=physics.dt,
        n_steps=physics.n_steps,
    )

    # Then
    assert fitted_ergotropy >= zero_ergotropy
