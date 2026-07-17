from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from quantum_battery_rl.benchmark import ppo_control
from quantum_battery_rl.benchmark.batch_dynamics import simulate_ensemble_pulse
from quantum_battery_rl.benchmark.controller import TrainingDraws, ValidationDraws
from quantum_battery_rl.benchmark.manifest import PhysicsConfig
from quantum_battery_rl.benchmark.ppo_control import (
    LegacyPolicyStateError,
    PpoControlConfig,
    PpoControlFitter,
    require_time_only_state_dimension,
)
from quantum_battery_rl.benchmark.uncertainty import DrawSet


DRAW_ROOT: Final = Path(__file__).parents[1] / "results" / "canonical" / "draws"


def _forbid_scalar(
    _pulse: NDArray[np.float64],
    **_parameters: float,
) -> tuple[NDArray[np.complex128], float]:
    raise AssertionError


def _draw_set(split: str, count: int) -> DrawSet:
    source = DrawSet.model_validate_json((DRAW_ROOT / f"{split}.json").read_bytes())
    return DrawSet(
        schema_version="1.0",
        split=source.split,
        source_manifest_sha256=source.source_manifest_sha256,
        draws=source.draws[:count],
    )


def _config() -> PpoControlConfig:
    return PpoControlConfig(
        physics=PhysicsConfig(
            t1=100.0,
            t2=80.0,
            omega_q=5.0,
            max_omega=0.25,
            n_steps=2,
            dt=0.1,
        ),
        manifest_sha256="a" * 64,
        implementation_sha256="b" * 64,
        environment_steps=8,
        checkpoint_interval_steps=4,
        learning_rate=0.001,
    )


def test_time_only_dimension_rejects_legacy_five_state_policy() -> None:
    require_time_only_state_dimension(1)
    with pytest.raises(LegacyPolicyStateError):
        require_time_only_state_dimension(5)


def test_ppo_fit_is_deterministic_frozen_and_hash_linked() -> None:
    # Given
    training = TrainingDraws(_draw_set("train", 2), "c" * 64)
    validation = ValidationDraws(_draw_set("validation", 2), "d" * 64)
    fitter = PpoControlFitter(_config(), validation)

    # When
    first = fitter.fit(training, seed=42)
    second = fitter.fit(training, seed=42)

    # Then
    assert first.metadata.method_id == "ppo-time-only-domain-randomized"
    assert first.controls == second.controls
    assert first.metadata.provenance.training_draws_sha256 == "c" * 64
    assert first.metadata.provenance.validation_draws_sha256 == "d" * 64
    assert tuple(item.draw_id for item in first.metadata.objective_contributions) == tuple(
        draw.draw_id for draw in training.draw_set.draws
    )
    assert first.metadata.cost.environment_steps >= 8
    assert first.pulse(2).shape == (2, 2)
    assert not first.pulse(2).flags.writeable


def test_ppo_fit_uses_batched_checkpoint_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    training = TrainingDraws(_draw_set("train", 2), "c" * 64)
    validation = ValidationDraws(_draw_set("validation", 2), "d" * 64)
    config = _config()
    fitter = PpoControlFitter(config, validation)
    monkeypatch.setattr(
        ppo_control,
        "simulate_pulse_sequence",
        _forbid_scalar,
        raising=False,
    )

    # When
    fitted = fitter.fit(training, seed=42)

    # Then
    observed = tuple(item.value for item in fitted.metadata.objective_contributions)
    expected = simulate_ensemble_pulse(
        fitted.pulse(config.physics.n_steps),
        training.draw_set.draws,
        config.physics,
    ).normalized_ergotropy
    assert np.max(np.abs(np.asarray(observed) - np.asarray(expected))) < 1e-10
