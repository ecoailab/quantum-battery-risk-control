"""Lindblad single-qubit battery environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TypedDict, cast, final

import numpy as np
from numpy.typing import NDArray

from .quantum_dynamics import (
    pauli_z,
    sigma_minus,
    build_hamiltonian,
    evolve_step,
    compute_energy,
    compute_ergotropy,
    bloch_vector,
)


class NonPhysicalCoherenceTimesError(ValueError):
    """Raised when coherence times are nonpositive or violate T2 <= 2*T1."""

    t1: float
    t2: float

    def __init__(self, t1: float, t2: float) -> None:
        self.t1 = t1
        self.t2 = t2
        message = (
            "Coherence times must be positive and T2 must not exceed 2*T1; "
            f"received T1={self.t1}, T2={self.t2}"
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EpisodeParameters:
    """Fixed coherence parameters for one complete episode."""

    t1: float
    t2: float

    def __post_init__(self) -> None:
        if self.t1 <= 0.0 or self.t2 <= 0.0 or self.t2 > 2.0 * self.t1:
            raise NonPhysicalCoherenceTimesError(t1=self.t1, t2=self.t2)


class InvalidActionShapeError(ValueError):
    """Raised when an action does not contain two quadrature values."""

    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        super().__init__(f"Action must have shape (2,); received {self.shape}")


class StepInfo(TypedDict):
    ergotropy: float
    energy: float
    step: int
    trace: float


NOISE_CONFIGS = {
    # --- Superconducting platforms (T1, T2 in microseconds) ---
    "Ideal": {"T1": 10000.0, "T2": 10000.0},
    "Low": {"T1": 500.0, "T2": 300.0},
    "IBM-like": {"T1": 100.0, "T2": 80.0},
    "High": {"T1": 50.0, "T2": 25.0},
    "Extreme": {"T1": 20.0, "T2": 10.0},
    # --- Trapped-ion platforms (converted to microseconds) ---
    "IonQ-Aria": {"T1": 10_000_000.0, "T2": 1_000_000.0},      # T1~10s, T2~1s
    "IonQ-Forte": {"T1": 100_000_000.0, "T2": 10_000_000.0},    # T1~100s, T2~10s
    "Quantinuum-H1": {"T1": 50_000_000.0, "T2": 3_000_000.0},   # T1~50s, T2~3s
    "Quantinuum-H2": {"T1": 100_000_000.0, "T2": 5_000_000.0},  # T1~100s, T2~5s
    # --- Future/projected platforms ---
    "IBM-Heron": {"T1": 300.0, "T2": 200.0},    # IBM Heron (projected)
    "NV-Center": {"T1": 6_000_000.0, "T2": 2_000.0},  # T1~6s, T2~2ms (room temp)
}


@final
class LindbladBatteryEnv:
    def __init__(
        self,
        T1: float = 100.0,
        T2: float = 80.0,
        omega_q: float = 5.0,
        max_omega: float = 0.25,
        n_steps: int = 100,
        dt: float = 0.1,
    ) -> None:
        self._episode_parameters = EpisodeParameters(t1=float(T1), t2=float(T2))
        self.omega_q = float(omega_q)
        self.max_omega = float(max_omega)
        self.n_steps = int(n_steps)
        self.dt = float(dt)

        self._omega_half = 0.5 * self.omega_q
        self._W_max = self.omega_q
        self._rng = np.random.default_rng()
        self.step_count = 0

        self.H0 = cast(
            NDArray[np.complex128],
            -0.5 * self.omega_q * pauli_z(),
        )
        self.gamma1 = 1.0 / self.T1 if self.T1 > 0.0 else 0.0
        self.gamma_phi = self._compute_gamma_phi()
        self.L_ops = self._build_lindblad_ops()

        self.rho = self._initial_state()

    @property
    def episode_parameters(self) -> EpisodeParameters:
        return self._episode_parameters

    @property
    def T1(self) -> float:
        return self._episode_parameters.t1

    @property
    def T2(self) -> float:
        return self._episode_parameters.t2

    def _initial_state(self) -> NDArray[np.complex128]:
        rho = np.zeros((2, 2), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def _compute_gamma_phi(self) -> float:
        if self.T2 <= 0.0:
            return 0.0
        if self.T1 > 0.0:
            gamma_phi = 0.5 * (1.0 / self.T2 - 0.5 / self.T1)
            return max(0.0, gamma_phi)
        return 0.5 / self.T2

    def _build_lindblad_ops(self) -> list[NDArray[np.complex128]]:
        ops: list[NDArray[np.complex128]] = []
        if self.gamma1 > 0.0:
            damping = sqrt(self.gamma1) * sigma_minus()
            ops.append(cast(NDArray[np.complex128], damping))
        if self.gamma_phi > 0.0:
            dephasing = sqrt(self.gamma_phi) * pauli_z()
            ops.append(cast(NDArray[np.complex128], dephasing))
        return ops

    def reset(self, seed: int | None = None) -> NDArray[np.float32]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.rho = self._initial_state()
        return self._get_observation()

    def _get_observation(self) -> NDArray[np.float32]:
        energy, ergotropy = self._compute_metrics()
        x, y, z = bloch_vector(self.rho)
        if self._omega_half == 0.0:
            energy_norm = 0.0
            ergotropy_norm = 0.0
        else:
            energy_norm = energy / self._omega_half
            ergotropy_norm = ergotropy / self._W_max
        return np.array([x, y, z, energy_norm, ergotropy_norm], dtype=np.float32)

    def _compute_metrics(self) -> tuple[float, float]:
        energy = compute_energy(self.rho, self.H0)
        ergotropy = compute_ergotropy(self.rho, self.H0)
        return energy, ergotropy

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[NDArray[np.float32], float, bool, bool, StepInfo]:
        action_array: NDArray[np.float64] = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)
        if action_array.shape[0] != 2:
            raise InvalidActionShapeError(shape=action_array.shape)
        clipped_action: NDArray[np.float64] = np.asarray(
            np.clip(action_array, -self.max_omega, self.max_omega),
            dtype=np.float64,
        )
        omega_x = float(cast(np.float64, clipped_action[0]))
        omega_y = float(cast(np.float64, clipped_action[1]))

        H = build_hamiltonian(self.omega_q, omega_x, omega_y)
        self.rho = cast(
            NDArray[np.complex128],
            evolve_step(self.rho, H, self.L_ops, self.dt),
        )
        self.step_count += 1

        energy, ergotropy = self._compute_metrics()
        if self._W_max == 0.0:
            ergotropy_norm = 0.0
        else:
            ergotropy_norm = ergotropy / self._W_max

        excited_population = float(cast(np.float64, self.rho[1, 1].real))
        control_cost = 0.01 * (omega_x**2 + omega_y**2)
        dissipation_cost = self.gamma1 * self.dt * excited_population
        reward = ergotropy_norm - control_cost - dissipation_cost

        done = self.step_count >= self.n_steps
        obs = self._get_observation()
        info: StepInfo = {
            "ergotropy": float(ergotropy_norm),
            "energy": float(energy),
            "step": int(self.step_count),
            "trace": self.get_trace(),
        }
        return obs, float(reward), done, False, info

    def get_ergotropy(self) -> float:
        _, ergotropy = self._compute_metrics()
        if self._W_max == 0.0:
            return 0.0
        return float(ergotropy / self._W_max)

    def get_state(self) -> NDArray[np.complex128]:
        return self.rho.copy()

    def get_trace(self) -> float:
        trace_real = cast(np.float64, (self.rho[0, 0] + self.rho[1, 1]).real)
        return float(trace_real)


if __name__ == "__main__":
    env = LindbladBatteryEnv(T1=10000, T2=10000)
    obs = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial ergotropy: {env.get_ergotropy():.6f}")

    total_reward = 0.0
    info = {"trace": env.get_trace()}
    for _ in range(100):
        action = np.array([0.25, 0.0])
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Final ergotropy (ideal): {env.get_ergotropy():.6f}")
    print(f"Trace preserved: {info['trace']:.10f}")

    env2 = LindbladBatteryEnv(T1=100, T2=80)
    obs2 = env2.reset(seed=42)
    for _ in range(100):
        action = np.array([0.25, 0.0])
        obs2, r, d, t, info2 = env2.step(action)
    print(f"Final ergotropy (IBM-like): {env2.get_ergotropy():.6f}")
