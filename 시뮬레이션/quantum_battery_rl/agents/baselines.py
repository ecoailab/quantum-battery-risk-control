from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from quantum_battery_rl.env.lindblad_env import LindbladBatteryEnv

from .base import (
    BaseAgent,
    InvalidPulseShapeError,
    MissingCoherenceTimesError,
    OptimizationEnvironment,
)
from .two_qubit_baselines import GRAPEController2Q, simulate_pulse_sequence_2q

__all__ = [
    "BangBangAgent",
    "GRAPEController",
    "GRAPEController2Q",
    "RandomAgent",
    "SinusoidalAgent",
    "simulate_pulse_sequence",
    "simulate_pulse_sequence_2q",
]


def simulate_pulse_sequence(
    pulse_sequence: np.ndarray,
    T1: float,
    T2: float,
    omega_q: float = 5.0,
    dt: float = 0.1,
    n_steps: int = 100,
) -> Tuple[np.ndarray, float]:
    pulses = np.asarray(pulse_sequence, dtype=float)
    if pulses.ndim != 2 or pulses.shape[1] != 2:
        raise InvalidPulseShapeError(shape=pulses.shape)
    n_steps = pulses.shape[0]
    largest_control = float(np.max(np.abs(pulses))) if pulses.size > 0 else 0.0
    environment = LindbladBatteryEnv(
        T1=T1,
        T2=T2,
        omega_q=omega_q,
        max_omega=max(0.25, largest_control),
        n_steps=n_steps,
        dt=dt,
    )
    _ = environment.reset()
    for action in pulses:
        _ = environment.step(action)
    return environment.get_state(), environment.get_ergotropy()


def _simulate_bangbang(n_on: float, amplitude: float, axis_angle: float,
                       T1: float, T2: float, omega_q: float, dt: float,
                       n_steps: int) -> float:
    pulse = np.zeros((n_steps, 2))
    n_on_int = int(round(n_on))
    n_on_int = max(0, min(n_on_int, n_steps))
    pulse[:n_on_int, 0] = amplitude * np.cos(axis_angle)
    pulse[:n_on_int, 1] = amplitude * np.sin(axis_angle)
    _, erg = simulate_pulse_sequence(pulse, T1, T2, omega_q, dt, n_steps)
    return erg


class BangBangAgent(BaseAgent):
    """Optimized bang-bang: constant max drive for N_on steps, then off.

    N_on is optimized via golden-section search per (T1, T2) pair.
    Axis angle is seed-dependent for per-seed variability.
    """

    def __init__(self, max_action: float = 0.25, n_steps: int = 100,
                 seed: Optional[int] = None, T1: float = 100.0,
                 T2: float = 80.0, omega_q: float = 5.0, dt: float = 0.1):
        super().__init__(max_action=max_action, n_steps=n_steps, seed=seed)
        self.T1 = T1
        self.T2 = T2
        self.omega_q = omega_q
        self.dt = dt
        self.action_sequence: Optional[np.ndarray] = None
        self._optimize_sequence()

    def _optimize_sequence(self) -> None:
        axis_angle = float(self.rng.uniform(0, 2 * np.pi))
        amp_noise = float(self.rng.uniform(0.85, 1.0))
        amplitude = self.max_action * amp_noise
        result: OptimizeResult = minimize_scalar(
            lambda n: -_simulate_bangbang(
                n, amplitude, axis_angle,
                self.T1, self.T2, self.omega_q, self.dt, self.n_steps),
            bounds=(1, self.n_steps),
            method="bounded",
            options={"xatol": 0.5},
        )
        n_on = int(round(result.x))
        self.action_sequence = np.zeros((self.n_steps, 2))
        self.action_sequence[:n_on, 0] = amplitude * np.cos(axis_angle)
        self.action_sequence[:n_on, 1] = amplitude * np.sin(axis_angle)

    def act(self, state: np.ndarray) -> np.ndarray:
        if self.action_sequence is None:
            self._optimize_sequence()
        assert self.action_sequence is not None
        index = min(self.step_counter, self.n_steps - 1)
        action = self.action_sequence[index]
        self.step_counter += 1
        return action.copy()

    def reset(self, seed: Optional[int] = None) -> None:
        seed_changed = seed is not None and seed != self.seed
        super().reset(seed=seed)
        if self.action_sequence is None or seed_changed:
            self._optimize_sequence()


class SinusoidalAgent(BaseAgent):
    """Sinusoidal drive with optimized frequency and seed-dependent phase.

    Frequency is optimized via bounded search; phase is seed-dependent.
    """

    def __init__(self, max_action: float = 0.25, n_steps: int = 100,
                 seed: Optional[int] = None, T1: float = 100.0,
                 T2: float = 80.0, omega_q: float = 5.0, dt: float = 0.1):
        super().__init__(max_action=max_action, n_steps=n_steps, seed=seed)
        self.T1 = T1
        self.T2 = T2
        self.omega_q = omega_q
        self.dt = dt
        self.omega: float = 0.0
        self.phase: float = 0.0
        self._optimize_params()

    def _build_pulse(self, omega: float, phase: float) -> np.ndarray:
        times = np.arange(self.n_steps) * self.dt
        amp = self.max_action * getattr(self, 'amp_scale', 1.0)
        pulse = np.zeros((self.n_steps, 2))
        pulse[:, 0] = amp * np.sin(omega * times + phase)
        pulse[:, 1] = amp * np.cos(omega * times + phase)
        return pulse

    def _optimize_params(self) -> None:
        self.phase = float(self.rng.uniform(0, 2 * np.pi))
        self.amp_scale = float(self.rng.uniform(0.85, 1.0))
        result: OptimizeResult = minimize_scalar(
            lambda w: -simulate_pulse_sequence(
                self._build_pulse(w, self.phase),
                self.T1, self.T2, self.omega_q, self.dt, self.n_steps)[1],
            bounds=(0.01, 5.0),
            method="bounded",
            options={"xatol": 0.01},
        )
        self.omega = float(result.x)

    def act(self, state: np.ndarray) -> np.ndarray:
        t = self.step_counter * self.dt
        amp = self.max_action * getattr(self, 'amp_scale', 1.0)
        action = np.array([
            amp * np.sin(self.omega * t + self.phase),
            amp * np.cos(self.omega * t + self.phase),
        ], dtype=float)
        self.step_counter += 1
        return action

    def reset(self, seed: Optional[int] = None) -> None:
        seed_changed = seed is not None and seed != self.seed
        super().reset(seed=seed)
        if seed_changed:
            self._optimize_params()


class RandomAgent(BaseAgent):
    """Uniform random control within bounds."""

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.rng.uniform(-self.max_action, self.max_action, size=2)
        self.step_counter += 1
        return action


class GRAPEController(BaseAgent):
    """GRAPE controller using L-BFGS-B (no explicit gradient — use approx_grad)."""

    def __init__(self, max_action: float = 0.25, n_steps: int = 100,
                 seed: Optional[int] = None, n_iters: int = 200,
                 omega_q: float = 5.0, dt: float = 0.1):
        super().__init__(max_action=max_action, n_steps=n_steps, seed=seed)
        self.n_iters = n_iters
        self.omega_q = omega_q
        self.dt = dt
        self.pulse_sequence: Optional[np.ndarray] = None
        self.optimized = False

    def _objective(self, flat_controls: np.ndarray, T1: float,
                   T2: float) -> float:
        pulse = flat_controls.reshape(self.n_steps, 2)
        pulse = np.clip(pulse, -self.max_action, self.max_action)
        _, ergotropy = simulate_pulse_sequence(
            pulse, T1=T1, T2=T2, omega_q=self.omega_q,
            dt=self.dt, n_steps=self.n_steps)
        return -ergotropy

    def optimize(
        self,
        T1: Optional[float] = None,
        T2: Optional[float] = None,
        omega_q: Optional[float] = None,
        dt: Optional[float] = None,
        env: OptimizationEnvironment | None = None,
    ) -> Tuple[float, np.ndarray]:
        t1 = T1
        t2 = T2
        if env is not None:
            if t1 is None:
                t1 = env.T1
            if t2 is None:
                t2 = env.T2
            if omega_q is None:
                omega_q = env.omega_q
            if dt is None:
                dt = env.dt

        if t1 is None or t2 is None:
            raise MissingCoherenceTimesError

        if omega_q is not None:
            self.omega_q = omega_q
        if dt is not None:
            self.dt = dt

        x0 = self.rng.uniform(-self.max_action, self.max_action,
                               size=self.n_steps * 2)
        bounds = [(-self.max_action, self.max_action)] * x0.size

        result: OptimizeResult = minimize(
            lambda x: self._objective(x, t1, t2),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.n_iters, "maxfun": self.n_iters * 5},
        )

        pulse = result.x.reshape(self.n_steps, 2)
        pulse = np.clip(pulse, -self.max_action, self.max_action)
        _, ergotropy = simulate_pulse_sequence(
            pulse, T1=t1, T2=t2, omega_q=self.omega_q,
            dt=self.dt, n_steps=self.n_steps)

        self.pulse_sequence = pulse
        self.optimized = True
        self.step_counter = 0
        return ergotropy, pulse

    def act(self, state: np.ndarray) -> np.ndarray:
        if self.pulse_sequence is None:
            return np.zeros(2, dtype=float)
        index = min(self.step_counter, self.n_steps - 1)
        action = self.pulse_sequence[index]
        self.step_counter += 1
        return action.copy()


if __name__ == "__main__":
    for AgentClass in [BangBangAgent, SinusoidalAgent, RandomAgent]:
        agent = AgentClass(seed=42)
        state = np.zeros(5, dtype=np.float32)
        action = agent.act(state)
        print(f"{AgentClass.__name__}: action={action}, shape={action.shape}")

    grape = GRAPEController(n_steps=100, n_iters=10, seed=42)
    erg, pulse = grape.optimize(T1=100.0, T2=80.0)
    print(f"GRAPE 1Q (10 iter): ergotropy={erg:.4f}, pulse shape={pulse.shape}")

    grape2q = GRAPEController2Q(n_steps=100, n_iters=10, seed=42)
    erg2q, pulse2q = grape2q.optimize(T1=100.0, T2=80.0)
    print(f"GRAPE 2Q (10 iter): ergotropy={erg2q:.4f}, pulse shape={pulse2q.shape}")
