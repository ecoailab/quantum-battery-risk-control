"""Canonical open-loop environment contracts."""

from __future__ import annotations

from typing import final

import numpy as np

from quantum_battery_rl.env.lindblad_env import (
    EpisodeParameters as EpisodeParameters,
    LindbladBatteryEnv,
    StepInfo,
)

from .manifest import PhysicsConfig


__all__ = ["CanonicalOpenLoopEnv", "EpisodeParameters"]


@final
class CanonicalOpenLoopEnv:
    def __init__(
        self,
        physics: PhysicsConfig,
        episode_parameters: EpisodeParameters,
    ) -> None:
        self._physics = physics
        self._episode_parameters = episode_parameters
        self._environment = LindbladBatteryEnv(
            T1=episode_parameters.t1,
            T2=episode_parameters.t2,
            omega_q=physics.omega_q,
            max_omega=physics.max_omega,
            n_steps=physics.n_steps,
            dt=physics.dt,
        )

    @property
    def episode_parameters(self) -> EpisodeParameters:
        return self._episode_parameters

    def reset(self, seed: int | None = None) -> np.ndarray:
        _ = self._environment.reset(seed=seed)
        return self._time_observation()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, StepInfo]:
        _, reward, done, truncated, info = self._environment.step(action)
        return self._time_observation(), reward, done, truncated, info

    def get_state(self) -> np.ndarray:
        return self._environment.get_state()

    def get_ergotropy(self) -> float:
        return self._environment.get_ergotropy()

    def _time_observation(self) -> np.ndarray:
        normalized_time = self._environment.step_count / self._physics.n_steps
        return np.array([normalized_time], dtype=np.float32)
