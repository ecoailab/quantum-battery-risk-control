"""Typed canonical controller fit and pulse contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from re import fullmatch
from typing import Annotated, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .uncertainty import DrawSet


NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonEmptyString = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class NonTrainingDrawSetError(ValueError):
    """Raised when fitting receives validation or test draws."""

    split: str

    def __init__(self, split: str) -> None:
        self.split = split
        super().__init__(f"Controller fitting requires train draws; received {split}")


class NonValidationDrawSetError(ValueError):
    """Raised when checkpoint selection receives train or test draws."""

    split: str

    def __init__(self, split: str) -> None:
        self.split = split
        super().__init__(f"Checkpoint selection requires validation draws; received {split}")


class InvalidProvenanceHashError(ValueError):
    """Raised when an input provenance hash is not SHA-256."""

    value: str

    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(f"Expected a lowercase SHA-256 hash; received {value!r}")


class PulseHorizonMismatchError(ValueError):
    """Raised when a frozen pulse is requested with another horizon."""

    expected: int
    received: int

    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        super().__init__(f"Pulse horizon is {expected}; received request for {received}")


@dataclass(frozen=True, slots=True)
class TrainingDraws:
    draw_set: DrawSet
    draw_set_sha256: str

    def __post_init__(self) -> None:
        if self.draw_set.split != "train":
            raise NonTrainingDrawSetError(self.draw_set.split)
        if fullmatch(r"[0-9a-f]{64}", self.draw_set_sha256) is None:
            raise InvalidProvenanceHashError(self.draw_set_sha256)


@dataclass(frozen=True, slots=True)
class ValidationDraws:
    draw_set: DrawSet
    draw_set_sha256: str

    def __post_init__(self) -> None:
        if self.draw_set.split != "validation":
            raise NonValidationDrawSetError(self.draw_set.split)
        if fullmatch(r"[0-9a-f]{64}", self.draw_set_sha256) is None:
            raise InvalidProvenanceHashError(self.draw_set_sha256)


class _FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class FitCost(_FrozenModel):
    objective_evaluations: NonNegativeInt
    gradient_evaluations: NonNegativeInt
    environment_steps: NonNegativeInt
    wall_time_seconds: NonNegativeFloat


class FitProvenance(_FrozenModel):
    manifest_sha256: Sha256
    training_draws_sha256: Sha256
    validation_draws_sha256: Sha256 | None
    implementation_sha256: Sha256
    optimizer_seed: int


class ObjectiveContribution(_FrozenModel):
    draw_id: NonEmptyString
    value: Annotated[float, Field(ge=0.0, le=1.0)]


class FitMetadata(_FrozenModel):
    method_id: NonEmptyString
    converged: bool
    termination_reason: NonEmptyString
    cost: FitCost
    provenance: FitProvenance
    objective_contributions: Annotated[
        tuple[ObjectiveContribution, ...],
        Field(min_length=1),
    ]


class FittedController(_FrozenModel):
    schema_version: Literal["1.0"]
    n_steps: PositiveInt
    max_action: PositiveFloat
    controls: Annotated[tuple[tuple[float, float], ...], Field(min_length=1)]
    metadata: FitMetadata

    @model_validator(mode="after")
    def validate_controls(self) -> Self:
        if len(self.controls) != self.n_steps:
            raise ValueError(
                f"Pulse contains {len(self.controls)} controls for {self.n_steps} steps",
            )
        if any(
            not isfinite(value) or abs(value) > self.max_action
            for control in self.controls
            for value in control
        ):
            raise ValueError("Pulse controls must be finite and within max_action")
        return self

    def pulse(self, n_steps: int) -> NDArray[np.float64]:
        if n_steps != self.n_steps:
            raise PulseHorizonMismatchError(self.n_steps, n_steps)
        pulse = np.asarray(self.controls, dtype=np.float64)
        pulse.setflags(write=False)
        return pulse


@runtime_checkable
class ControllerFitter(Protocol):
    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController: ...


@runtime_checkable
class PulseController(Protocol):
    @property
    def metadata(self) -> FitMetadata: ...

    def pulse(self, n_steps: int) -> NDArray[np.float64]: ...
