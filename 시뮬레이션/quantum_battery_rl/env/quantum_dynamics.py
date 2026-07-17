"""Quantum dynamics utilities for single-qubit and two-qubit systems."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_SIGMA_PLUS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


def pauli_x() -> np.ndarray:
    return _PAULI_X.copy()


def pauli_y() -> np.ndarray:
    return _PAULI_Y.copy()


def pauli_z() -> np.ndarray:
    return _PAULI_Z.copy()


def sigma_plus() -> np.ndarray:
    return _SIGMA_PLUS.copy()


def sigma_minus() -> np.ndarray:
    return _SIGMA_MINUS.copy()


def build_hamiltonian(omega_q: float, omega_x: float, omega_y: float) -> np.ndarray:
    return -0.5 * omega_q * _PAULI_Z + omega_x * _PAULI_X + omega_y * _PAULI_Y


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, L_ops: list[np.ndarray]) -> np.ndarray:
    commutator = -1j * (H @ rho - rho @ H)
    dissipator = np.zeros_like(rho, dtype=np.complex128)
    for L in L_ops:
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        dissipator += L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
    return commutator + dissipator


def evolve_step(
    rho: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    dt: float,
) -> np.ndarray:
    dim = rho.shape[0]
    ident = np.eye(dim, dtype=np.complex128)
    liouvillian = -1j * (np.kron(ident, H) - np.kron(H.T, ident))
    for L in L_ops:
        L_dag_L = L.conj().T @ L
        liouvillian += np.kron(L.conj(), L) - 0.5 * (
            np.kron(ident, L_dag_L) + np.kron(L_dag_L.T, ident)
        )
    vec_rho = rho.reshape(dim * dim, order="F")
    vec_rho = expm(liouvillian * dt) @ vec_rho
    rho_next = vec_rho.reshape((dim, dim), order="F")
    return ensure_physical(rho_next)


def ensure_physical(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    trace = float(np.sum(eigenvalues))
    if trace == 0.0:
        rho_fixed = np.zeros_like(rho, dtype=np.complex128)
        rho_fixed[0, 0] = 1.0
        return rho_fixed
    eigenvalues /= trace
    rho_fixed = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    return rho_fixed


def compute_ergotropy(rho: np.ndarray, H: np.ndarray) -> float:
    energy = compute_energy(rho, H)
    rho_eigs = np.linalg.eigvalsh(rho)
    rho_eigs_desc = rho_eigs[::-1]
    H_eigs_asc = np.linalg.eigvalsh(H)
    passive_energy = float(np.sum(rho_eigs_desc * H_eigs_asc))
    ergotropy = energy - passive_energy
    if ergotropy < 0.0 and np.isclose(ergotropy, 0.0, atol=1e-12):
        ergotropy = 0.0
    return float(max(ergotropy, 0.0))


def compute_energy(rho: np.ndarray, H: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ H)))


def bloch_vector(rho: np.ndarray) -> tuple[float, float, float]:
    x = float(np.real(np.trace(_PAULI_X @ rho)))
    y = float(np.real(np.trace(_PAULI_Y @ rho)))
    z = float(np.real(np.trace(_PAULI_Z @ rho)))
    return x, y, z


# ---------------------------------------------------------------------------
# Two-qubit utilities
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=np.complex128)


def _kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


def build_H0_2q(omega_q: float) -> np.ndarray:
    """Return the two-qubit free Hamiltonian with |00> as the ground state."""
    return -0.5 * omega_q * (_kron(_PAULI_Z, _I2) + _kron(_I2, _PAULI_Z))


def build_drive_hamiltonian_2q(
    omega_x1: float, omega_y1: float,
    omega_x2: float, omega_y2: float,
) -> np.ndarray:
    """Rotating-frame drive: Ωx1·(σx⊗I) + Ωy1·(σy⊗I) + Ωx2·(I⊗σx) + Ωy2·(I⊗σy)."""
    H = (omega_x1 * _kron(_PAULI_X, _I2)
         + omega_y1 * _kron(_PAULI_Y, _I2)
         + omega_x2 * _kron(_I2, _PAULI_X)
         + omega_y2 * _kron(_I2, _PAULI_Y))
    return H


def build_lindblad_ops_2q(gamma1: float, gamma_phi: float) -> list[np.ndarray]:
    """Local Lindblad operators on each qubit (no correlated noise).

    L1a = √γ1 · (σ⁻⊗I),  L1b = √γ1 · (I⊗σ⁻)
    L2a = √γφ · (σz⊗I),  L2b = √γφ · (I⊗σz)
    """
    ops: list[np.ndarray] = []
    if gamma1 > 0.0:
        ops.append(np.sqrt(gamma1) * _kron(_SIGMA_MINUS, _I2))
        ops.append(np.sqrt(gamma1) * _kron(_I2, _SIGMA_MINUS))
    if gamma_phi > 0.0:
        ops.append(np.sqrt(gamma_phi) * _kron(_PAULI_Z, _I2))
        ops.append(np.sqrt(gamma_phi) * _kron(_I2, _PAULI_Z))
    return ops


def ground_state_2q() -> np.ndarray:
    """Return both qubits in the ground state |00><00|."""
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho
