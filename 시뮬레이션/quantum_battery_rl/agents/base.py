"""Shared contracts for open-loop baseline controllers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class OptimizationEnvironment(Protocol):
    T1: float
    T2: float
    omega_q: float
    dt: float


class InvalidPulseShapeError(ValueError):
    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        super().__init__(shape)

    def __str__(self) -> str:
        return f"Pulse sequence must have shape (n_steps, 2); received {self.shape}"


class MissingCoherenceTimesError(ValueError):
    def __str__(self) -> str:
        return "T1 and T2 are required"


class BaseAgent:
    """Mutable controller state shared by deterministic and random baselines."""

    def __init__(
        self,
        max_action: float = 0.25,
        n_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        self.max_action = max_action
        self.n_steps = n_steps
        self.seed = seed
        self.step_counter = 0
        self.rng = np.random.default_rng(seed)

    def act(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.step_counter = 0
