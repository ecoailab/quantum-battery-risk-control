from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence
from quantum_battery_rl.benchmark import cvar_control, heuristic_control, mean_control
from quantum_battery_rl.benchmark.batch_dynamics import simulate_ensemble_pulse
from quantum_battery_rl.benchmark.controller import TrainingDraws
from quantum_battery_rl.benchmark.cvar_control import CvarControlConfig, CvarControlFitter
from quantum_battery_rl.benchmark.heuristic_control import (
    HeuristicControlConfig,
    RandomReferenceFitter,
)
from quantum_battery_rl.benchmark.manifest import PhysicsConfig, load_manifest
from quantum_battery_rl.benchmark.mean_control import MeanControlConfig, MeanControlFitter
from quantum_battery_rl.benchmark.uncertainty import DrawSet, UncertaintyDraw


PROJECT_ROOT: Final = Path(__file__).parents[1]
TRAIN_PATH: Final = PROJECT_ROOT / "results" / "canonical" / "draws" / "train.json"


def _physics() -> PhysicsConfig:
    source = load_manifest(PROJECT_ROOT / "canonical_manifest.json").physics
    return source.model_copy(update={"n_steps": 8})


def _canonical_physics() -> PhysicsConfig:
    return load_manifest(PROJECT_ROOT / "canonical_manifest.json").physics


def _draws() -> tuple[UncertaintyDraw, ...]:
    source = DrawSet.model_validate_json(TRAIN_PATH.read_bytes()).draws
    return tuple(source[index] for index in (0, 127, 128, 255, 256, 383))


def _training_draws() -> TrainingDraws:
    source = DrawSet.model_validate_json(TRAIN_PATH.read_bytes())
    selected = DrawSet(
        schema_version="1.0",
        split="train",
        source_manifest_sha256=source.source_manifest_sha256,
        draws=_draws(),
    )
    return TrainingDraws(selected, "d" * 64)


def _forbid_scalar(
    _pulse: NDArray[np.float64],
    **_parameters: float,
) -> tuple[NDArray[np.complex128], float]:
    raise AssertionError


def test_batched_ensemble_matches_scalar_states_and_values() -> None:
    # Given
    physics = _physics()
    draws = _draws()
    random = np.random.default_rng(20260716)

    for _ in range(5):
        pulse = random.uniform(
            -physics.max_omega,
            physics.max_omega,
            size=(physics.n_steps, 2),
        )

        # When
        batched = simulate_ensemble_pulse(pulse, draws, physics)
        scalar = tuple(
            simulate_pulse_sequence(
                pulse,
                T1=draw.t1,
                T2=draw.t2,
                omega_q=physics.omega_q,
                dt=physics.dt,
                n_steps=physics.n_steps,
            )
            for draw in draws
        )

        # Then
        scalar_states = np.stack(tuple(state for state, _ in scalar))
        scalar_values = np.asarray(tuple(value for _, value in scalar))
        assert np.max(np.abs(batched.states - scalar_states)) < 1e-10
        assert np.max(
            np.abs(np.asarray(batched.normalized_ergotropy) - scalar_values),
        ) < 1e-10
        assert not batched.states.flags.writeable


def test_batched_ensemble_preserves_density_matrix_invariants() -> None:
    # Given
    physics = _physics()
    draws = _draws()
    pulse = np.full((physics.n_steps, 2), physics.max_omega)

    # When
    result = simulate_ensemble_pulse(pulse, draws, physics)

    # Then
    traces: np.ndarray[tuple[int], np.dtype[np.complex128]] = (
        result.states[:, 0, 0] + result.states[:, 1, 1]
    )
    eigenvalues = np.linalg.eigvalsh(result.states)
    assert np.max(np.abs(traces - 1.0)) < 1e-12
    assert np.max(np.abs(result.states - result.states.conj().transpose(0, 2, 1))) < 1e-12
    assert np.min(eigenvalues) >= -1e-12
    assert all(0.0 <= value <= 1.0 for value in result.normalized_ergotropy)


def test_batched_ensemble_matches_scalar_at_full_horizon() -> None:
    # Given
    physics = _canonical_physics()
    draws = _draws()
    random = np.random.default_rng(20260717)
    pulses = (
        np.zeros((physics.n_steps, 2), dtype=np.float64),
        np.full((physics.n_steps, 2), physics.max_omega),
        random.uniform(
            -physics.max_omega,
            physics.max_omega,
            size=(physics.n_steps, 2),
        ),
    )

    for pulse in pulses:
        # When
        batched = simulate_ensemble_pulse(pulse, draws, physics)
        scalar = tuple(
            simulate_pulse_sequence(
                pulse,
                T1=draw.t1,
                T2=draw.t2,
                omega_q=physics.omega_q,
                dt=physics.dt,
                n_steps=physics.n_steps,
            )
            for draw in draws
        )

        # Then
        scalar_states = np.stack(tuple(state for state, _ in scalar))
        scalar_values = np.asarray(tuple(value for _, value in scalar))
        assert np.max(np.abs(batched.states - scalar_states)) < 1e-10
        assert np.max(
            np.abs(np.asarray(batched.normalized_ergotropy) - scalar_values),
        ) < 1e-10


def test_mean_objective_uses_batched_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    physics = _physics()
    training = _training_draws()
    fitter = MeanControlFitter(
        MeanControlConfig(
            physics=physics,
            manifest_sha256="a" * 64,
            implementation_sha256="b" * 64,
            max_iterations=1,
            max_objective_evaluations=20,
        ),
    )
    monkeypatch.setattr(
        mean_control,
        "simulate_pulse_sequence",
        _forbid_scalar,
        raising=False,
    )

    # When
    fitted = fitter.fit(training, seed=42)

    # Then
    observed = tuple(item.value for item in fitted.metadata.objective_contributions)
    expected = simulate_ensemble_pulse(
        fitted.pulse(physics.n_steps),
        training.draw_set.draws,
        physics,
    ).normalized_ergotropy
    assert np.max(np.abs(np.asarray(observed) - np.asarray(expected))) < 1e-10


def test_cvar_objective_uses_batched_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    physics = _physics()
    training = _training_draws()
    fitter = CvarControlFitter(
        CvarControlConfig(
            physics=physics,
            manifest_sha256="a" * 64,
            implementation_sha256="b" * 64,
            max_iterations=1,
            max_objective_evaluations=20,
        ),
    )
    monkeypatch.setattr(
        cvar_control,
        "simulate_pulse_sequence",
        _forbid_scalar,
        raising=False,
    )

    # When
    fitted = fitter.fit(training, seed=42)

    # Then
    observed = fitted.metadata.objective_contributions
    expected = simulate_ensemble_pulse(
        fitted.pulse(physics.n_steps),
        training.draw_set.draws,
        physics,
    ).normalized_ergotropy
    assert tuple(item.draw_id for item in observed) == tuple(
        draw.draw_id for draw in training.draw_set.draws
    )
    assert np.max(
        np.abs(
            np.asarray(tuple(item.value for item in observed)) - np.asarray(expected),
        ),
    ) < 1e-10


def test_heuristic_evaluation_uses_batched_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    physics = _physics()
    training = _training_draws()
    fitter = RandomReferenceFitter(
        HeuristicControlConfig(
            physics=physics,
            manifest_sha256="a" * 64,
            implementation_sha256="b" * 64,
            axis_count=2,
            frequency_count=2,
            phase_count=2,
        ),
    )
    monkeypatch.setattr(
        heuristic_control,
        "simulate_pulse_sequence",
        _forbid_scalar,
        raising=False,
    )

    # When
    fitted = fitter.fit(training, seed=42)

    # Then
    observed = tuple(item.value for item in fitted.metadata.objective_contributions)
    expected = simulate_ensemble_pulse(
        fitted.pulse(physics.n_steps),
        training.draw_set.draws,
        physics,
    ).normalized_ergotropy
    assert np.max(np.abs(np.asarray(observed) - np.asarray(expected))) < 1e-10
