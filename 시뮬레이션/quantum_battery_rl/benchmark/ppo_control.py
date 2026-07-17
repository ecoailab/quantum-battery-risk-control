"""Canonical time-only domain-randomized PPO control."""

from __future__ import annotations

from importlib import import_module
from time import perf_counter
from typing import Annotated, ClassVar, Protocol, TypedDict, cast, final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from quantum_battery_rl.agents.ppo import PPOAgent, set_seed

from .batch_dynamics import simulate_ensemble_pulse
from .controller import (
    FitCost,
    FitMetadata,
    FitProvenance,
    FittedController,
    ObjectiveContribution,
    TrainingDraws,
    ValidationDraws,
)
from .environment import CanonicalOpenLoopEnv, EpisodeParameters
from .manifest import PhysicsConfig
from .uncertainty import UncertaintyDraw


PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Trajectory(TypedDict):
    states: NDArray[np.float32]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    dones: NDArray[np.float32]
    log_probs: NDArray[np.float32]
    values: NDArray[np.float32]
    next_state: NDArray[np.float32]


class _CollectTrajectories(Protocol):
    def __call__(
        self,
        agent: PPOAgent,
        environments: list[CanonicalOpenLoopEnv],
        n_steps: int,
    ) -> list[_Trajectory]: ...


class _UpdateAgent(Protocol):
    def __call__(
        self,
        trajectories: list[_Trajectory],
    ) -> dict[str, float]: ...


_collect_trajectories = cast(
    _CollectTrajectories,
    getattr(import_module("quantum_battery_rl.agents.ppo"), "collect_trajectories"),
)


class LegacyPolicyStateError(ValueError):
    """Raised when a checkpoint expects the legacy five-value state."""

    state_dimension: int

    def __init__(self, state_dimension: int) -> None:
        self.state_dimension = state_dimension
        super().__init__(f"Canonical PPO requires state_dim=1; received {state_dimension}")


def require_time_only_state_dimension(state_dimension: int) -> None:
    if state_dimension != 1:
        raise LegacyPolicyStateError(state_dimension)


class PpoControlConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    physics: PhysicsConfig
    manifest_sha256: Sha256
    implementation_sha256: Sha256
    environment_steps: PositiveInt
    checkpoint_interval_steps: PositiveInt
    learning_rate: PositiveFloat


@final
class PpoControlFitter:
    _config: PpoControlConfig
    _validation_draws: ValidationDraws

    def __init__(
        self,
        config: PpoControlConfig,
        validation_draws: ValidationDraws,
    ) -> None:
        self._config = config
        self._validation_draws = validation_draws

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        require_time_only_state_dimension(1)
        set_seed(seed)
        physics = self._config.physics
        agent = PPOAgent(
            state_dim=1,
            action_dim=2,
            max_action=physics.max_omega,
            lr=self._config.learning_rate,
            n_envs=1,
            ppo_epochs=2,
            batch_size=physics.n_steps,
            device="cpu",
        )
        random = np.random.default_rng(seed)
        training_steps = 0
        update_count = 0
        validation_evaluations = 0
        best_score = float("-inf")
        best_pulse = np.zeros((physics.n_steps, 2), dtype=np.float64)
        started = perf_counter()
        draws = training_draws.draw_set.draws
        while training_steps < self._config.environment_steps:
            draw = draws[int(random.integers(0, len(draws)))]
            environment = CanonicalOpenLoopEnv(
                physics,
                EpisodeParameters(t1=draw.t1, t2=draw.t2),
            )
            trajectories = _collect_trajectories(
                agent,
                [environment],
                n_steps=physics.n_steps,
            )
            update_agent = cast(_UpdateAgent, agent.update)
            _ = update_agent(trajectories)
            training_steps += physics.n_steps
            update_count += 1
            should_select = (
                training_steps % self._config.checkpoint_interval_steps == 0
                or training_steps >= self._config.environment_steps
            )
            if should_select:
                candidate = _deterministic_pulse(agent, physics)
                values = _evaluate_pulse(
                    candidate,
                    self._validation_draws.draw_set.draws,
                    physics,
                )
                validation_evaluations += len(values)
                score = sum(values) / len(values)
                if score > best_score:
                    best_score = score
                    best_pulse = candidate
        training_values = _evaluate_pulse(best_pulse, draws, physics)
        elapsed = perf_counter() - started
        controls = tuple(
            (
                float(cast(np.float64, best_pulse[index, 0])),
                float(cast(np.float64, best_pulse[index, 1])),
            )
            for index in range(physics.n_steps)
        )
        metadata = FitMetadata(
            method_id="ppo-time-only-domain-randomized",
            converged=True,
            termination_reason="budget-complete-validation-selected",
            cost=FitCost(
                objective_evaluations=validation_evaluations,
                gradient_evaluations=update_count,
                environment_steps=(
                    training_steps
                    + validation_evaluations * physics.n_steps
                    + len(draws) * physics.n_steps
                ),
                wall_time_seconds=elapsed,
            ),
            provenance=FitProvenance(
                manifest_sha256=self._config.manifest_sha256,
                training_draws_sha256=training_draws.draw_set_sha256,
                validation_draws_sha256=self._validation_draws.draw_set_sha256,
                implementation_sha256=self._config.implementation_sha256,
                optimizer_seed=seed,
            ),
            objective_contributions=tuple(
                ObjectiveContribution(draw_id=draw.draw_id, value=value)
                for draw, value in zip(draws, training_values, strict=True)
            ),
        )
        return FittedController(
            schema_version="1.0",
            n_steps=physics.n_steps,
            max_action=physics.max_omega,
            controls=controls,
            metadata=metadata,
        )


def _deterministic_pulse(
    agent: PPOAgent,
    physics: PhysicsConfig,
) -> NDArray[np.float64]:
    pulse = np.zeros((physics.n_steps, 2), dtype=np.float64)
    for index in range(physics.n_steps):
        observation = np.array([index / physics.n_steps], dtype=np.float32)
        action, _, _ = agent.act(observation, deterministic=True)
        pulse[index] = action
    return pulse


def _evaluate_pulse(
    pulse: NDArray[np.float64],
    draws: tuple[UncertaintyDraw, ...],
    physics: PhysicsConfig,
) -> tuple[float, ...]:
    return simulate_ensemble_pulse(pulse, draws, physics).normalized_ergotropy
