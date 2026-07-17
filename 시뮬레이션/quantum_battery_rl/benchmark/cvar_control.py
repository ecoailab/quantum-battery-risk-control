"""Canonical lower-tail CVaR pulse optimization."""

from __future__ import annotations

from importlib import import_module
from math import ceil
from time import perf_counter
from typing import Annotated, Callable, ClassVar, Protocol, cast, final

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


class _OptimizeResult(Protocol):
    x: NDArray[np.float64]
    success: bool
    message: str
    njev: int


class _Minimize(Protocol):
    def __call__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        initial: NDArray[np.float64],
        *,
        method: str,
        bounds: list[tuple[float, float]],
        options: dict[str, int],
    ) -> _OptimizeResult: ...


_minimize = cast(_Minimize, getattr(import_module("scipy.optimize"), "minimize"))


class CvarControlConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    physics: PhysicsConfig
    manifest_sha256: Sha256
    implementation_sha256: Sha256
    max_iterations: PositiveInt
    max_objective_evaluations: PositiveInt


def select_lower_tail(
    contributions: tuple[ObjectiveContribution, ...],
) -> tuple[ObjectiveContribution, ...]:
    tail_count = max(1, ceil(0.1 * len(contributions)))
    ordered = sorted(contributions, key=lambda item: (item.value, item.draw_id))
    return tuple(ordered[:tail_count])


@final
class _CvarObjective:
    evaluations: int

    def __init__(
        self,
        physics: PhysicsConfig,
        draws: tuple[UncertaintyDraw, ...],
    ) -> None:
        self._physics = physics
        self._draws = draws
        self.evaluations = 0

    def contributions(
        self,
        flat_controls: NDArray[np.float64],
    ) -> tuple[ObjectiveContribution, ...]:
        self.evaluations += 1
        pulse = flat_controls.reshape(self._physics.n_steps, 2)
        values = simulate_ensemble_pulse(
            pulse,
            self._draws,
            self._physics,
        ).normalized_ergotropy
        return tuple(
            ObjectiveContribution(
                draw_id=draw.draw_id,
                value=value,
            )
            for draw, value in zip(self._draws, values, strict=True)
        )

    def __call__(self, flat_controls: NDArray[np.float64]) -> float:
        tail = select_lower_tail(self.contributions(flat_controls))
        return -sum(item.value for item in tail) / len(tail)


@final
class CvarControlFitter:
    _config: CvarControlConfig

    def __init__(self, config: CvarControlConfig) -> None:
        self._config = config

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        physics = self._config.physics
        draws = training_draws.draw_set.draws
        random = np.random.default_rng(seed)
        initial = random.uniform(
            -physics.max_omega,
            physics.max_omega,
            size=physics.n_steps * 2,
        )
        bounds = [(-physics.max_omega, physics.max_omega)] * initial.size
        objective = _CvarObjective(physics, draws)
        started = perf_counter()
        result = _minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self._config.max_iterations,
                "maxfun": self._config.max_objective_evaluations,
            },
        )
        elapsed = perf_counter() - started
        pulse = cast(
            NDArray[np.float64],
            np.asarray(
                np.clip(result.x, -physics.max_omega, physics.max_omega),
                dtype=np.float64,
            ).reshape(physics.n_steps, 2),
        )
        contributions = objective.contributions(pulse.reshape(-1))
        controls = tuple(
            (
                float(cast(np.float64, pulse[index, 0])),
                float(cast(np.float64, pulse[index, 1])),
            )
            for index in range(physics.n_steps)
        )
        metadata = FitMetadata(
            method_id="cvar-0.1-lbfgsb-finite-difference",
            converged=result.success,
            termination_reason=result.message,
            cost=FitCost(
                objective_evaluations=objective.evaluations,
                gradient_evaluations=result.njev,
                environment_steps=(
                    objective.evaluations * physics.n_steps * len(draws)
                ),
                wall_time_seconds=elapsed,
            ),
            provenance=FitProvenance(
                manifest_sha256=self._config.manifest_sha256,
                training_draws_sha256=training_draws.draw_set_sha256,
                validation_draws_sha256=None,
                implementation_sha256=self._config.implementation_sha256,
                optimizer_seed=seed,
            ),
            objective_contributions=contributions,
        )
        return FittedController(
            schema_version="1.0",
            n_steps=physics.n_steps,
            max_action=physics.max_omega,
            controls=controls,
            metadata=metadata,
        )
