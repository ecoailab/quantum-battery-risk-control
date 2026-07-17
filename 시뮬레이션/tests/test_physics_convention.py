from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quantum_battery_rl.agents.baselines import (
    GRAPEController2Q,
    simulate_pulse_sequence,
    simulate_pulse_sequence_2q,
)
from quantum_battery_rl.env.lindblad_env import LindbladBatteryEnv
from quantum_battery_rl.env.quantum_dynamics import (
    build_hamiltonian,
    bloch_vector,
    build_H0_2q,
    build_lindblad_ops_2q,
    compute_energy,
    compute_ergotropy,
    evolve_step,
    ground_state_2q,
    sigma_minus,
)


def _basis_density(index: int) -> NDArray[np.complex128]:
    density = np.zeros((2, 2), dtype=np.complex128)
    density[index, index] = 1.0
    return density


def test_baseline_and_environment_start_from_same_density_matrix() -> None:
    # Given
    environment = LindbladBatteryEnv(T1=1.0e12, T2=1.0e12, n_steps=1)
    zero_pulse = np.zeros((1, 2), dtype=np.float64)

    # When
    environment_state = environment.get_state()
    baseline_state, _ = simulate_pulse_sequence(
        zero_pulse,
        T1=1.0e12,
        T2=1.0e12,
        n_steps=1,
    )

    # Then
    np.testing.assert_allclose(environment_state, baseline_state, atol=1.0e-12)


@pytest.mark.parametrize(
    ("t1", "t2"),
    [(100.0, 80.0), (50.0, 25.0), (500.0, 300.0)],
)
def test_baseline_and_environment_match_for_bounded_pulses(
    t1: float,
    t2: float,
) -> None:
    # Given
    random = np.random.default_rng(20260716)

    for _ in range(20):
        pulses = random.uniform(-0.25, 0.25, size=(8, 2))
        environment = LindbladBatteryEnv(T1=t1, T2=t2, n_steps=8, dt=0.1)
        _ = environment.reset(seed=42)

        # When
        baseline_state, baseline_ergotropy = simulate_pulse_sequence(
            pulses,
            T1=t1,
            T2=t2,
            n_steps=8,
            dt=0.1,
        )
        for action in pulses:
            _ = environment.step(action)

        # Then
        np.testing.assert_allclose(
            baseline_state,
            environment.get_state(),
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            baseline_ergotropy,
            environment.get_ergotropy(),
            atol=1.0e-10,
        )


def test_two_qubit_baseline_symbols_remain_usable() -> None:
    # Given
    controller = GRAPEController2Q(n_steps=1, n_iters=1, seed=42)
    zero_pulse = np.zeros((1, 2), dtype=np.float64)

    # When
    state, ergotropy = simulate_pulse_sequence_2q(
        zero_pulse,
        T1=100.0,
        T2=80.0,
        n_steps=1,
    )

    # Then
    assert controller.n_steps == 1
    assert state.shape == (4, 4)
    assert np.isfinite(ergotropy)


def test_zero_state_has_lower_energy_than_one_state() -> None:
    # Given
    hamiltonian = build_hamiltonian(5.0, 0.0, 0.0)

    # When
    zero_energy = compute_energy(_basis_density(0), hamiltonian)
    one_energy = compute_energy(_basis_density(1), hamiltonian)

    # Then
    assert zero_energy < one_energy


def test_zero_state_is_passive() -> None:
    # Given
    hamiltonian = build_hamiltonian(5.0, 0.0, 0.0)

    # When
    ergotropy = compute_ergotropy(_basis_density(0), hamiltonian)

    # Then
    np.testing.assert_allclose(ergotropy, 0.0, atol=1.0e-12)


def test_one_state_has_unit_normalized_ergotropy() -> None:
    # Given
    omega_q = 5.0
    hamiltonian = build_hamiltonian(omega_q, 0.0, 0.0)

    # When
    normalized_ergotropy = compute_ergotropy(
        _basis_density(1),
        hamiltonian,
    ) / omega_q

    # Then
    np.testing.assert_allclose(normalized_ergotropy, 1.0, atol=1.0e-12)


def test_sigma_minus_maps_excited_state_to_ground_state() -> None:
    # Given
    excited_ket = np.array([0.0, 1.0], dtype=np.complex128)
    ground_ket = np.array([1.0, 0.0], dtype=np.complex128)

    # When
    lowered_ket = sigma_minus() @ excited_ket

    # Then
    np.testing.assert_allclose(lowered_ket, ground_ket, atol=1.0e-12)


def test_amplitude_damping_reduces_excited_population() -> None:
    # Given
    excited_density = _basis_density(1)
    zero_hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    damping_operators = [np.sqrt(0.1) * sigma_minus()]

    # When
    evolved_density = evolve_step(
        excited_density,
        zero_hamiltonian,
        damping_operators,
        1.0,
    )

    # Then
    assert float(np.real(evolved_density[1, 1])) < 1.0
    assert float(np.real(evolved_density[0, 0])) > 0.0


def test_zero_drive_ground_state_remains_passive() -> None:
    # Given
    environment = LindbladBatteryEnv(T1=100.0, T2=80.0, n_steps=1, dt=0.1)
    environment.reset(seed=42)

    # When
    initial_ergotropy = environment.get_ergotropy()
    _, reward, _, _, _ = environment.step(np.zeros(2, dtype=np.float64))
    final_ergotropy = environment.get_ergotropy()

    # Then
    np.testing.assert_allclose(
        [initial_ergotropy, final_ergotropy],
        [0.0, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(reward, 0.0, atol=1.0e-12)


def test_ground_state_has_unit_positive_bloch_z() -> None:
    # Given
    ground_density = _basis_density(0)

    # When
    vector = bloch_vector(ground_density)

    # Then
    np.testing.assert_allclose(vector, (0.0, 0.0, 1.0), atol=1.0e-12)


def test_nonphysical_coherence_times_are_rejected() -> None:
    # Given
    t1 = 10.0
    t2 = 20.1

    # When / Then
    with pytest.raises(ValueError):
        _ = LindbladBatteryEnv(T1=t1, T2=t2)


def test_invalid_action_error_records_received_shape() -> None:
    # Given
    environment = LindbladBatteryEnv(n_steps=1)
    invalid_action = np.zeros(3, dtype=np.float64)

    # When
    with pytest.raises(ValueError) as error_info:
        _ = environment.step(invalid_action)

    # Then
    assert getattr(error_info.value, "shape", None) == (3,)


def test_noiseless_pi_pulse_charges_ground_state() -> None:
    # Given
    drive_amplitude = 0.25
    pulse_duration = np.pi / (2.0 * drive_amplitude)
    drive_hamiltonian = build_hamiltonian(0.0, drive_amplitude, 0.0)

    # When
    charged_density = evolve_step(
        _basis_density(0),
        drive_hamiltonian,
        [],
        pulse_duration,
    )

    # Then
    np.testing.assert_allclose(charged_density, _basis_density(1), atol=1.0e-10)


def test_two_qubit_ground_state_uses_zero_zero_basis_state() -> None:
    # Given
    expected_ground = np.zeros((4, 4), dtype=np.complex128)
    expected_ground[0, 0] = 1.0

    # When
    actual_ground = ground_state_2q()
    ground_energy = compute_energy(actual_ground, build_H0_2q(5.0))

    # Then
    np.testing.assert_allclose(actual_ground, expected_ground, atol=1.0e-12)
    np.testing.assert_allclose(ground_energy, -5.0, atol=1.0e-12)


def test_two_qubit_local_damping_lowers_single_excitation() -> None:
    # Given
    excited_density = np.zeros((4, 4), dtype=np.complex128)
    excited_density[1, 1] = 1.0
    zero_hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    damping_operators = build_lindblad_ops_2q(gamma1=0.1, gamma_phi=0.0)

    # When
    evolved_density = evolve_step(
        excited_density,
        zero_hamiltonian,
        damping_operators,
        1.0,
    )

    # Then
    assert float(np.real(evolved_density[1, 1])) < 1.0
    assert float(np.real(evolved_density[0, 0])) > 0.0
