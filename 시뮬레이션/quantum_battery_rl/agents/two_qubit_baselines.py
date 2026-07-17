"""Legacy two-qubit baseline compatibility on shared dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from quantum_battery_rl.env.quantum_dynamics import (
    build_H0_2q,
    build_drive_hamiltonian_2q,
    build_lindblad_ops_2q,
    compute_ergotropy,
    evolve_step,
    ground_state_2q,
)

from .base import (
    BaseAgent,
    InvalidPulseShapeError,
    MissingCoherenceTimesError,
    OptimizationEnvironment,
)


def simulate_pulse_sequence_2q(
    pulse_sequence: NDArray[np.floating],
    T1: float,
    T2: float,
    omega_q: float = 5.0,
    dt: float = 0.1,
    n_steps: int = 100,
) -> tuple[NDArray[np.complex128], float]:
    """Simulate identical local drives on two qubits using shared dynamics."""
    pulses = np.asarray(pulse_sequence, dtype=np.float64)
    if pulses.ndim != 2 or pulses.shape[1] != 2:
        raise InvalidPulseShapeError(shape=pulses.shape)

    actual_steps = pulses.shape[0]
    if actual_steps != n_steps:
        n_steps = actual_steps

    gamma_1 = 1.0 / T1 if T1 > 0.0 else 0.0
    gamma_phi = max(0.0, 1.0 / T2 - 0.5 * gamma_1) if T2 > 0.0 else 0.0
    lindblad_operators = build_lindblad_ops_2q(gamma_1, gamma_phi)
    density = ground_state_2q()

    for step in range(n_steps):
        omega_x, omega_y = pulses[step]
        drive = build_drive_hamiltonian_2q(
            float(omega_x),
            float(omega_y),
            float(omega_x),
            float(omega_y),
        )
        density = evolve_step(density, drive, lindblad_operators, dt)

    ergotropy = compute_ergotropy(density, build_H0_2q(omega_q))
    normalized_ergotropy = ergotropy / omega_q if omega_q != 0.0 else 0.0
    return density, float(normalized_ergotropy)


class GRAPEController2Q(BaseAgent):
    """Finite-difference L-BFGS-B controller for the legacy two-qubit path."""

    def __init__(
        self,
        max_action: float = 0.25,
        n_steps: int = 100,
        seed: int | None = None,
        n_iters: int = 200,
        omega_q: float = 5.0,
        dt: float = 0.1,
    ) -> None:
        super().__init__(max_action=max_action, n_steps=n_steps, seed=seed)
        self.n_iters = n_iters
        self.omega_q = omega_q
        self.dt = dt
        self.pulse_sequence: NDArray[np.float64] | None = None
        self.optimized = False

    def _objective(self, flat_controls: NDArray[np.floating], t1: float, t2: float) -> float:
        pulse = np.clip(
            flat_controls.reshape(self.n_steps, 2),
            -self.max_action,
            self.max_action,
        )
        _, ergotropy = simulate_pulse_sequence_2q(
            pulse,
            T1=t1,
            T2=t2,
            omega_q=self.omega_q,
            dt=self.dt,
            n_steps=self.n_steps,
        )
        return -ergotropy

    def optimize(
        self,
        T1: float | None = None,
        T2: float | None = None,
        omega_q: float | None = None,
        dt: float | None = None,
        env: OptimizationEnvironment | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        t1 = T1
        t2 = T2
        if env is not None:
            t1 = env.T1 if t1 is None else t1
            t2 = env.T2 if t2 is None else t2
            omega_q = env.omega_q if omega_q is None else omega_q
            dt = env.dt if dt is None else dt
        if t1 is None or t2 is None:
            raise MissingCoherenceTimesError
        if omega_q is not None:
            self.omega_q = omega_q
        if dt is not None:
            self.dt = dt

        initial = self.rng.uniform(
            -self.max_action,
            self.max_action,
            size=self.n_steps * 2,
        )
        result: OptimizeResult = minimize(
            lambda controls: self._objective(controls, t1, t2),
            initial,
            method="L-BFGS-B",
            bounds=[(-self.max_action, self.max_action)] * initial.size,
            options={"maxiter": self.n_iters, "maxfun": self.n_iters * 5},
        )
        pulse = np.clip(
            result.x.reshape(self.n_steps, 2),
            -self.max_action,
            self.max_action,
        )
        _, ergotropy = simulate_pulse_sequence_2q(
            pulse,
            T1=t1,
            T2=t2,
            omega_q=self.omega_q,
            dt=self.dt,
            n_steps=self.n_steps,
        )
        self.pulse_sequence = pulse
        self.optimized = True
        self.step_counter = 0
        return ergotropy, pulse

    def act(self, state: NDArray[np.floating]) -> NDArray[np.float64]:
        if self.pulse_sequence is None:
            return np.zeros(2, dtype=np.float64)
        index = min(self.step_counter, self.n_steps - 1)
        action = self.pulse_sequence[index]
        self.step_counter += 1
        return action.copy()
