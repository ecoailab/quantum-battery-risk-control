from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from quantum_battery_rl.benchmark import CanonicalOpenLoopEnv, EpisodeParameters
from quantum_battery_rl.benchmark.manifest import load_manifest
from quantum_battery_rl.env.lindblad_env import (
    LindbladBatteryEnv,
    NonPhysicalCoherenceTimesError,
)


MANIFEST_PATH: Final = Path(__file__).parents[1] / "canonical_manifest.json"


def test_episode_parameters_are_frozen_and_physical() -> None:
    # Given
    parameters = EpisodeParameters(t1=100.0, t2=80.0)

    # When / Then
    with pytest.raises(FrozenInstanceError):
        setattr(parameters, "t1", 50.0)
    with pytest.raises(NonPhysicalCoherenceTimesError):
        _ = EpisodeParameters(t1=10.0, t2=20.1)


def test_legacy_environment_coherence_times_are_read_only() -> None:
    # Given
    environment = LindbladBatteryEnv(T1=100.0, T2=80.0)

    # When / Then
    with pytest.raises(AttributeError):
        setattr(environment, "T1", 50.0)
    assert environment.T1 == 100.0
    assert environment.T2 == 80.0


def test_canonical_observation_contains_only_normalized_time() -> None:
    # Given
    physics = load_manifest(MANIFEST_PATH).physics
    environment = CanonicalOpenLoopEnv(
        physics,
        EpisodeParameters(t1=100.0, t2=80.0),
    )
    zero_action = np.zeros(2, dtype=np.float64)

    # When
    observations = [environment.reset(seed=42)]
    for _ in range(physics.n_steps):
        observation, _, _, _, _ = environment.step(zero_action)
        observations.append(observation)

    # Then
    assert all(observation.shape == (1,) for observation in observations)
    assert all(observation.dtype == np.float32 for observation in observations)
    np.testing.assert_allclose(observations[0], [0.0], atol=0.0)
    np.testing.assert_allclose(observations[1], [1.0 / physics.n_steps])
    np.testing.assert_allclose(observations[-1], [1.0], atol=0.0)


def test_observations_do_not_reveal_state_or_episode_parameters() -> None:
    # Given
    physics = load_manifest(MANIFEST_PATH).physics
    first = CanonicalOpenLoopEnv(
        physics,
        EpisodeParameters(t1=100.0, t2=80.0),
    )
    second = CanonicalOpenLoopEnv(
        physics,
        EpisodeParameters(t1=50.0, t2=25.0),
    )
    action = np.array([physics.max_omega, 0.0], dtype=np.float64)

    # When
    first_observations = [first.reset(seed=1)]
    second_observations = [second.reset(seed=2)]
    for _ in range(10):
        first_observation, _, _, _, _ = first.step(action)
        second_observation, _, _, _, _ = second.step(action)
        first_observations.append(first_observation)
        second_observations.append(second_observation)

    # Then
    for first_observation, second_observation in zip(
        first_observations,
        second_observations,
        strict=True,
    ):
        np.testing.assert_array_equal(first_observation, second_observation)
    assert not np.allclose(first.get_state(), second.get_state(), atol=1.0e-12)


def test_episode_parameter_identity_is_constant_through_reset_and_steps() -> None:
    # Given
    physics = load_manifest(MANIFEST_PATH).physics
    parameters = EpisodeParameters(t1=100.0, t2=80.0)
    environment = CanonicalOpenLoopEnv(physics, parameters)
    zero_action = np.zeros(2, dtype=np.float64)

    # When
    _ = environment.reset(seed=42)
    for _ in range(physics.n_steps):
        _ = environment.step(zero_action)

    # Then
    assert environment.episode_parameters is parameters
