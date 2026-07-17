"""Canonical low-complexity heuristic control fitters."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, ClassVar, cast, final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .batch_dynamics import simulate_ensemble_pulse
from .controller import (
    FitCost,
    FitMetadata,
    FitProvenance,
    FittedController,
    ObjectiveContribution,
    TrainingDraws,
)
from .manifest import PhysicsConfig
from .uncertainty import UncertaintyDraw


PositiveInt = Annotated[int, Field(gt=0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class HeuristicControlConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    physics: PhysicsConfig
    manifest_sha256: Sha256
    implementation_sha256: Sha256
    axis_count: PositiveInt
    frequency_count: PositiveInt
    phase_count: PositiveInt


@dataclass(frozen=True, slots=True)
class _FitRequest:
    config: HeuristicControlConfig
    training_draws: TrainingDraws
    seed: int
    method_id: str
    pulse: NDArray[np.float64]
    values: tuple[float, ...]
    objective_evaluations: int
    wall_time_seconds: float


@final
class BangBangFitter:
    _config: HeuristicControlConfig

    def __init__(self, config: HeuristicControlConfig) -> None:
        self._config = config

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        started = perf_counter()
        physics = self._config.physics
        best_score = float("-inf")
        best_pulse = np.zeros((physics.n_steps, 2), dtype=np.float64)
        best_values: tuple[float, ...] = ()
        for duration in range(1, physics.n_steps + 1):
            for axis_index in range(self._config.axis_count):
                angle = 2.0 * np.pi * axis_index / self._config.axis_count
                pulse = np.zeros((physics.n_steps, 2), dtype=np.float64)
                pulse[:duration, 0] = physics.max_omega * np.cos(angle)
                pulse[:duration, 1] = physics.max_omega * np.sin(angle)
                values = _evaluate(pulse, training_draws.draw_set.draws, physics)
                score = sum(values) / len(values)
                if score > best_score:
                    best_score = score
                    best_pulse = pulse
                    best_values = values
        return _fitted(
            _FitRequest(
                config=self._config,
                training_draws=training_draws,
                seed=seed,
                method_id="bang-bang-ensemble-grid",
                pulse=best_pulse,
                values=best_values,
                objective_evaluations=physics.n_steps * self._config.axis_count,
                wall_time_seconds=perf_counter() - started,
            ),
        )


@final
class SinusoidalFitter:
    _config: HeuristicControlConfig

    def __init__(self, config: HeuristicControlConfig) -> None:
        self._config = config

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        started = perf_counter()
        physics = self._config.physics
        frequencies = np.linspace(0.01, physics.omega_q, self._config.frequency_count)
        phases = np.linspace(0.0, 2.0 * np.pi, self._config.phase_count, endpoint=False)
        times = np.arange(physics.n_steps, dtype=np.float64) * physics.dt
        best_score = float("-inf")
        best_pulse = np.zeros((physics.n_steps, 2), dtype=np.float64)
        best_values: tuple[float, ...] = ()
        for frequency in frequencies:
            for phase in phases:
                pulse = np.zeros((physics.n_steps, 2), dtype=np.float64)
                pulse[:, 0] = physics.max_omega * np.sin(frequency * times + phase)
                pulse[:, 1] = physics.max_omega * np.cos(frequency * times + phase)
                values = _evaluate(pulse, training_draws.draw_set.draws, physics)
                score = sum(values) / len(values)
                if score > best_score:
                    best_score = score
                    best_pulse = pulse
                    best_values = values
        return _fitted(
            _FitRequest(
                config=self._config,
                training_draws=training_draws,
                seed=seed,
                method_id="sinusoidal-ensemble-grid",
                pulse=best_pulse,
                values=best_values,
                objective_evaluations=(
                    self._config.frequency_count * self._config.phase_count
                ),
                wall_time_seconds=perf_counter() - started,
            ),
        )


@final
class RandomReferenceFitter:
    _config: HeuristicControlConfig

    def __init__(self, config: HeuristicControlConfig) -> None:
        self._config = config

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        started = perf_counter()
        physics = self._config.physics
        random = np.random.default_rng(seed)
        pulse = random.uniform(
            -physics.max_omega,
            physics.max_omega,
            size=(physics.n_steps, 2),
        )
        values = _evaluate(pulse, training_draws.draw_set.draws, physics)
        return _fitted(
            _FitRequest(
                config=self._config,
                training_draws=training_draws,
                seed=seed,
                method_id="random-seeded-reference",
                pulse=pulse,
                values=values,
                objective_evaluations=1,
                wall_time_seconds=perf_counter() - started,
            ),
        )


def _evaluate(
    pulse: NDArray[np.float64],
    draws: tuple[UncertaintyDraw, ...],
    physics: PhysicsConfig,
) -> tuple[float, ...]:
    return simulate_ensemble_pulse(pulse, draws, physics).normalized_ergotropy


def _fitted(request: _FitRequest) -> FittedController:
    physics = request.config.physics
    controls = tuple(
        (
            float(cast(np.float64, request.pulse[index, 0])),
            float(cast(np.float64, request.pulse[index, 1])),
        )
        for index in range(physics.n_steps)
    )
    metadata = FitMetadata(
        method_id=request.method_id,
        converged=True,
        termination_reason="candidate-search-complete",
        cost=FitCost(
            objective_evaluations=request.objective_evaluations,
            gradient_evaluations=0,
            environment_steps=(
                request.objective_evaluations
                * physics.n_steps
                * len(request.training_draws.draw_set.draws)
            ),
            wall_time_seconds=request.wall_time_seconds,
        ),
        provenance=FitProvenance(
            manifest_sha256=request.config.manifest_sha256,
            training_draws_sha256=request.training_draws.draw_set_sha256,
            validation_draws_sha256=None,
            implementation_sha256=request.config.implementation_sha256,
            optimizer_seed=request.seed,
        ),
        objective_contributions=tuple(
            ObjectiveContribution(draw_id=draw.draw_id, value=value)
            for draw, value in zip(
                request.training_draws.draw_set.draws,
                request.values,
                strict=True,
            )
        ),
    )
    return FittedController(
        schema_version="1.0",
        n_steps=physics.n_steps,
        max_action=physics.max_omega,
        controls=controls,
        metadata=metadata,
    )
