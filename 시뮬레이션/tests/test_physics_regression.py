from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_battery_rl.env.lindblad_env import LindbladBatteryEnv
from quantum_battery_rl.env.quantum_dynamics import evolve_step


def _basis_density(index: int) -> NDArray[np.complex128]:
    density = np.zeros((2, 2), dtype=np.complex128)
    density[index, index] = 1.0
    return density


def _plus_density() -> NDArray[np.complex128]:
    return np.full((2, 2), 0.5, dtype=np.complex128)


def test_amplitude_damping_matches_exponential_population_decay() -> None:
    # Given
    t1 = 10.0
    dt = 0.5
    step_count = 4
    environment = LindbladBatteryEnv(T1=t1, T2=2.0 * t1)
    density = _basis_density(1)

    # When
    for _ in range(step_count):
        density = evolve_step(density, environment.H0, environment.L_ops, dt)

    # Then
    expected_population = np.exp(-(step_count * dt) / t1)
    np.testing.assert_allclose(
        float(np.real(density[1, 1])),
        expected_population,
        atol=1.0e-12,
    )


def test_dephasing_matches_declared_t2_coherence_decay() -> None:
    # Given
    t1 = 1.0e12
    t2 = 10.0
    dt = 1.0
    environment = LindbladBatteryEnv(T1=t1, T2=t2)

    # When
    density = evolve_step(
        _plus_density(),
        np.zeros((2, 2), dtype=np.complex128),
        environment.L_ops,
        dt,
    )

    # Then
    expected_coherence = 0.5 * np.exp(-dt / t2)
    np.testing.assert_allclose(
        abs(density[0, 1]),
        expected_coherence,
        atol=1.0e-12,
    )


def test_dephasing_preserves_populations_when_relaxation_is_negligible() -> None:
    # Given
    environment = LindbladBatteryEnv(T1=1.0e12, T2=10.0)

    # When
    density = evolve_step(
        _plus_density(),
        np.zeros((2, 2), dtype=np.complex128),
        environment.L_ops,
        1.0,
    )

    # Then
    np.testing.assert_allclose(np.diag(density), [0.5, 0.5], atol=1.0e-12)


def test_bounded_drive_preserves_density_matrix_invariants() -> None:
    # Given
    environment = LindbladBatteryEnv(T1=100.0, T2=80.0, n_steps=50)
    random = np.random.default_rng(20260716)

    # When / Then
    for action in random.uniform(-0.25, 0.25, size=(50, 2)):
        _, _, _, _, info = environment.step(action)
        density = environment.get_state()
        eigenvalues = np.linalg.eigvalsh(density)
        np.testing.assert_allclose(info["trace"], 1.0, atol=1.0e-12)
        np.testing.assert_allclose(density, density.conj().T, atol=1.0e-12)
        assert float(np.min(eigenvalues)) >= -1.0e-12
        assert 0.0 <= environment.get_ergotropy() <= 1.0
