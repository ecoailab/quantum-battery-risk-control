"""Strict canonical experiment manifest boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ManifestConstraintError(ValueError):
    """Base class for typed cross-field manifest failures."""


class NonPhysicalManifestError(ManifestConstraintError):
    t1: float
    t2: float

    def __init__(self, t1: float, t2: float) -> None:
        self.t1 = t1
        self.t2 = t2
        super().__init__(t1, t2)

    def __str__(self) -> str:
        return f"Manifest requires T2 <= 2*T1; received T1={self.t1}, T2={self.t2}"


class DuplicateSeedError(ManifestConstraintError):
    seeds: tuple[int, ...]

    def __init__(self, seeds: tuple[int, ...]) -> None:
        self.seeds = seeds
        super().__init__(seeds)

    def __str__(self) -> str:
        return f"Seeds must be unique; received {self.seeds}"


class IncreasingSequenceError(ManifestConstraintError):
    values: tuple[float | int, ...]

    def __init__(self, values: tuple[float | int, ...]) -> None:
        self.values = values
        super().__init__(values)

    def __str__(self) -> str:
        return f"Values must be strictly increasing; received {self.values}"


class StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


PositiveFloat = Annotated[float, Field(gt=0.0)]
UnitFraction = Annotated[float, Field(gt=0.0, lt=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class PhysicsConfig(StrictManifestModel):
    t1: PositiveFloat
    t2: PositiveFloat
    omega_q: PositiveFloat
    max_omega: PositiveFloat
    n_steps: PositiveInt
    dt: PositiveFloat

    @model_validator(mode="after")
    def verify_coherence_times(self) -> Self:
        if self.t2 > 2.0 * self.t1:
            raise NonPhysicalManifestError(t1=self.t1, t2=self.t2)
        return self


class UncertaintyConfig(StrictManifestModel):
    source: Literal["synthetic_joint_stress_test"]
    severity_fractions: Annotated[tuple[UnitFraction, ...], Field(min_length=1)]
    differential_ratio: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def verify_severity_order(self) -> Self:
        if tuple(sorted(set(self.severity_fractions))) != self.severity_fractions:
            raise IncreasingSequenceError(values=self.severity_fractions)
        return self


class SplitSeeds(StrictManifestModel):
    train: int
    validation: int
    test: int

    @model_validator(mode="after")
    def verify_disjoint(self) -> Self:
        seeds = (self.train, self.validation, self.test)
        if len(set(seeds)) != len(seeds):
            raise DuplicateSeedError(seeds=seeds)
        return self


class ScenarioCounts(StrictManifestModel):
    train: PositiveInt
    validation: PositiveInt
    test: PositiveInt


class SplitConfig(StrictManifestModel):
    seeds: SplitSeeds
    scenarios: ScenarioCounts


class OptimizationConfig(StrictManifestModel):
    optimizer_seeds: Annotated[tuple[int, ...], Field(min_length=1)]
    grape_evaluation_stages: Annotated[tuple[PositiveInt, ...], Field(min_length=1)]
    ppo_environment_step_stages: Annotated[tuple[PositiveInt, ...], Field(min_length=1)]
    smoke_scenarios: PositiveInt

    @model_validator(mode="after")
    def verify_sequences(self) -> Self:
        if len(set(self.optimizer_seeds)) != len(self.optimizer_seeds):
            raise DuplicateSeedError(seeds=self.optimizer_seeds)
        for values in (
            self.grape_evaluation_stages,
            self.ppo_environment_step_stages,
        ):
            if tuple(sorted(set(values))) != values:
                raise IncreasingSequenceError(values=values)
        return self


class StatisticsConfig(StrictManifestModel):
    cvar_alpha: UnitFraction
    bootstrap_samples: PositiveInt
    q_threshold: UnitFraction
    cliff_threshold: UnitFraction
    confidence_level: UnitFraction


class OutputConfig(StrictManifestModel):
    root: Literal["results/canonical"]


class CanonicalManifest(StrictManifestModel):
    schema_version: Literal["1.0"]
    physics: PhysicsConfig
    uncertainty: UncertaintyConfig
    splits: SplitConfig
    optimization: OptimizationConfig
    statistics: StatisticsConfig
    output: OutputConfig


def load_manifest(path: Path) -> CanonicalManifest:
    return CanonicalManifest.model_validate_json(path.read_bytes())
