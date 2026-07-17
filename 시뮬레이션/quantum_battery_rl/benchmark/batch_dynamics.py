"""Batched canonical one-qubit ensemble evolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from quantum_battery_rl.agents.base import InvalidPulseShapeError
from quantum_battery_rl.env.quantum_dynamics import (
    build_hamiltonian,
    compute_ergotropy,
    pauli_z,
    sigma_minus,
)

from .manifest import PhysicsConfig
from .uncertainty import UncertaintyDraw


@dataclass(frozen=True, slots=True)
class EnsembleSimulation:
    states: NDArray[np.complex128]
    normalized_ergotropy: tuple[float, ...]


class EmptyEnsembleError(ValueError):
    """Raised when ensemble evolution receives no uncertainty draws."""


def simulate_ensemble_pulse(
    pulse: NDArray[np.float64],
    draws: tuple[UncertaintyDraw, ...],
    physics: PhysicsConfig,
) -> EnsembleSimulation:
    pulses: NDArray[np.float64] = np.asarray(pulse, dtype=np.float64)
    if pulses.shape != (physics.n_steps, 2):
        raise InvalidPulseShapeError(shape=pulses.shape)
    if not draws:
        raise EmptyEnsembleError
    identity = np.eye(2, dtype=np.complex128)
    damping = _dissipator_superoperator(sigma_minus(), identity)
    dephasing = _dissipator_superoperator(pauli_z(), identity)
    gamma_1 = np.asarray(tuple(1.0 / draw.t1 for draw in draws), dtype=np.float64)
    gamma_phi = np.asarray(
        tuple(0.5 * (1.0 / draw.t2 - 0.5 / draw.t1) for draw in draws),
        dtype=np.float64,
    )
    vectorized_states = np.zeros((len(draws), 4), dtype=np.complex128)
    vectorized_states[:, 0] = 1.0
    for step in range(physics.n_steps):
        hamiltonian = build_hamiltonian(
            physics.omega_q,
            float(pulses[step : step + 1, 0:1].reshape(())),
            float(pulses[step : step + 1, 1:2].reshape(())),
        )
        coherent = -1j * (
            np.kron(identity, hamiltonian)
            - np.kron(hamiltonian.T, identity)
        )
        liouvillians = (
            coherent[np.newaxis, :, :]
            + gamma_1[:, np.newaxis, np.newaxis] * damping[np.newaxis, :, :]
            + gamma_phi[:, np.newaxis, np.newaxis] * dephasing[np.newaxis, :, :]
        )
        eigenvalues: NDArray[np.complex128]
        eigenvectors: NDArray[np.complex128]
        eigenvalues, eigenvectors = np.linalg.eig(liouvillians)
        coordinates: NDArray[np.complex128] = np.linalg.solve(
            eigenvectors,
            vectorized_states[:, :, np.newaxis],
        )[:, :, 0]
        coordinates *= np.exp(eigenvalues * physics.dt)
        evolved = np.empty_like(vectorized_states)
        for row in range(4):
            evolved[:, row] = (
                eigenvectors[:, row, 0] * coordinates[:, 0]
                + eigenvectors[:, row, 1] * coordinates[:, 1]
                + eigenvectors[:, row, 2] * coordinates[:, 2]
                + eigenvectors[:, row, 3] * coordinates[:, 3]
            )
        vectorized_states = evolved
        states: NDArray[np.complex128] = np.reshape(
            vectorized_states,
            (-1, 2, 2),
            order="F",
        )
        states = _ensure_physical_batch(states)
        vectorized_states = np.reshape(states, (-1, 4), order="F")
    states = np.reshape(vectorized_states, (-1, 2, 2), order="F")
    states.setflags(write=False)
    hamiltonian_0 = build_hamiltonian(physics.omega_q, 0.0, 0.0)
    values = tuple(
        compute_ergotropy(
            states[index : index + 1].reshape((2, 2)),
            hamiltonian_0,
        )
        / physics.omega_q
        for index in range(len(draws))
    )
    return EnsembleSimulation(states=states, normalized_ergotropy=values)


def _dissipator_superoperator(
    operator: NDArray[np.complex128],
    identity: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    product = operator.conj().T @ operator
    return np.kron(operator.conj(), operator) - 0.5 * (
        np.kron(identity, product) + np.kron(product.T, identity)
    )


def _ensure_physical_batch(
    states: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    hermitian = 0.5 * (states + states.conj().transpose(0, 2, 1))
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped: NDArray[np.float64] = np.clip(eigenvalues, 0.0, None)
    traces: NDArray[np.float64] = clipped[:, 0] + clipped[:, 1]
    normalized: NDArray[np.float64] = clipped / traces[:, np.newaxis]
    fixed = (eigenvectors * normalized[:, np.newaxis, :]) @ eigenvectors.conj().transpose(
        0,
        2,
        1,
    )
    return fixed
