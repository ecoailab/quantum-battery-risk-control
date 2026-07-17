from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence
from quantum_battery_rl.benchmark.controller import ObjectiveContribution, TrainingDraws
from quantum_battery_rl.benchmark.cvar_control import (
    CvarControlConfig,
    CvarControlFitter,
    select_lower_tail,
)
from quantum_battery_rl.benchmark.manifest import PhysicsConfig
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
        draws=source.draws[:11],
    )
    return TrainingDraws(draw_set, "d" * 64)


def _config() -> CvarControlConfig:
    return CvarControlConfig(
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


def test_lower_tail_uses_fixed_ten_percent_and_draw_id_tie_break() -> None:
    contributions = (
        ObjectiveContribution(draw_id="b", value=0.1),
        ObjectiveContribution(draw_id="a", value=0.1),
        *(ObjectiveContribution(draw_id=f"z{i}", value=0.8) for i in range(9)),
    )

    selected = select_lower_tail(contributions)

    assert tuple(item.draw_id for item in selected) == ("a", "b")


def test_cvar_fitter_records_all_draws_and_optimizes_tail_mean() -> None:
    # Given
    training = _training_draws()
    fitter = CvarControlFitter(_config())
    zero = np.zeros((2, 2), dtype=np.float64)

    # When
    first = fitter.fit(training, seed=42)
    second = fitter.fit(training, seed=42)
    tail = select_lower_tail(first.metadata.objective_contributions)
    zero_contributions = tuple(
        ObjectiveContribution(
            draw_id=draw.draw_id,
            value=_ergotropy(zero, draw.t1, draw.t2),
        )
        for draw in training.draw_set.draws
    )

    # Then
    assert first.metadata.method_id == "cvar-0.1-lbfgsb-finite-difference"
    assert first.metadata.converged
    assert first.controls == second.controls
    assert len(first.metadata.objective_contributions) == 11
    assert len(tail) == 2
    assert sum(item.value for item in tail) / len(tail) >= (
        sum(item.value for item in select_lower_tail(zero_contributions))
        / len(select_lower_tail(zero_contributions))
    )
    assert first.metadata.provenance.training_draws_sha256 == "d" * 64
    assert first.metadata.cost.environment_steps == (
        first.metadata.cost.objective_evaluations * 2 * 11
    )
