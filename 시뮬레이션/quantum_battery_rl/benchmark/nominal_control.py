"""Canonical nominal fixed-calibration pulse optimization."""

from __future__ import annotations

from importlib import import_module
from time import perf_counter
from typing import Annotated, Callable, ClassVar, Protocol, cast, final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from quantum_battery_rl.agents.baselines import simulate_pulse_sequence

from .controller import (
    FitCost,
    FitMetadata,
    FitProvenance,
    FittedController,
    ObjectiveContribution,
    TrainingDraws,
)
from .manifest import PhysicsConfig


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


class NominalControlConfig(BaseModel):
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


@final
class _NominalObjective:
    evaluations: int

    def __init__(self, physics: PhysicsConfig) -> None:
        self._physics = physics
        self.evaluations = 0

    def __call__(self, flat_controls: NDArray[np.float64]) -> float:
        self.evaluations += 1
        pulse = np.clip(
            flat_controls.reshape(self._physics.n_steps, 2),
            -self._physics.max_omega,
            self._physics.max_omega,
        )
        _, ergotropy = simulate_pulse_sequence(
            pulse,
            T1=self._physics.t1,
            T2=self._physics.t2,
            omega_q=self._physics.omega_q,
            dt=self._physics.dt,
            n_steps=self._physics.n_steps,
        )
        return -ergotropy


@final
class NominalControlFitter:
    _config: NominalControlConfig

    def __init__(self, config: NominalControlConfig) -> None:
        self._config = config

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        physics = self._config.physics
        random = np.random.default_rng(seed)
        initial = random.uniform(
            -physics.max_omega,
            physics.max_omega,
            size=physics.n_steps * 2,
        )
        bounds = [(-physics.max_omega, physics.max_omega)] * initial.size
        objective = _NominalObjective(physics)
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
        controls = tuple(
            (
                float(cast(np.float64, pulse[index, 0])),
                float(cast(np.float64, pulse[index, 1])),
            )
            for index in range(physics.n_steps)
        )
        final_ergotropy = -objective(pulse.reshape(-1))
        metadata = FitMetadata(
            method_id="nominal-lbfgsb-finite-difference",
            converged=result.success,
            termination_reason=result.message,
            cost=FitCost(
                objective_evaluations=objective.evaluations,
                gradient_evaluations=result.njev,
                environment_steps=objective.evaluations * physics.n_steps,
                wall_time_seconds=elapsed,
            ),
            provenance=FitProvenance(
                manifest_sha256=self._config.manifest_sha256,
                training_draws_sha256=training_draws.draw_set_sha256,
                validation_draws_sha256=None,
                implementation_sha256=self._config.implementation_sha256,
                optimizer_seed=seed,
            ),
            objective_contributions=(
                ObjectiveContribution(draw_id="nominal", value=final_ergotropy),
            ),
        )
        return FittedController(
            schema_version="1.0",
            n_steps=physics.n_steps,
            max_action=physics.max_omega,
            controls=controls,
            metadata=metadata,
        )
