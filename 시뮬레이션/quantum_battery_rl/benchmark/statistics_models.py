"""Strict seed-level statistical summary artifacts."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .controller import FitCost


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
UnitValue = Annotated[float, Field(ge=0.0, le=1.0)]
UnitFraction = Annotated[float, Field(gt=0.0, lt=1.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
EffectSize = Annotated[float, Field(ge=-1.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonEmptyString = Annotated[str, Field(min_length=1)]


class InvalidIntervalError(ValueError):
    """Raised when an estimate lies outside its confidence interval."""


class DuplicateStatisticsEntryError(ValueError):
    """Raised when a statistical artifact repeats an identity."""


class _FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class MetricValues(_FrozenModel):
    mean: UnitValue
    median: UnitValue
    percentile_10: UnitValue
    cvar_0_1: UnitValue
    minimum: UnitValue


class EstimateInterval(_FrozenModel):
    estimate: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def verify_order(self) -> Self:
        if not self.lower <= self.estimate <= self.upper:
            raise InvalidIntervalError
        return self


class MetricIntervals(_FrozenModel):
    mean: EstimateInterval
    median: EstimateInterval
    percentile_10: EstimateInterval
    cvar_0_1: EstimateInterval
    minimum: EstimateInterval


class SeedStatistics(_FrozenModel):
    method_id: NonEmptyString
    optimizer_seed: int
    metrics: MetricValues


class SeverityStatistics(_FrozenModel):
    severity_fraction: UnitFraction
    metrics: MetricIntervals


class MethodStatistics(_FrozenModel):
    method_id: NonEmptyString
    seed_count: PositiveInt
    metrics: MetricIntervals
    severity: Annotated[tuple[SeverityStatistics, ...], Field(min_length=1)]
    fit_cost_total: FitCost


class PairwiseComparison(_FrozenModel):
    method_a: NonEmptyString
    method_b: NonEmptyString
    seed_pairs: PositiveInt
    mean_difference: EstimateInterval
    wilcoxon_p: Probability
    bh_q: Probability
    cliffs_delta: EffectSize


class StatisticsArtifact(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["statistics"]
    manifest_sha256: Sha256
    raw_results_sha256: Sha256
    raw_results_receipt_sha256: Sha256
    statistics_implementation_sha256: Sha256
    bootstrap_seed: int
    bootstrap_samples: PositiveInt
    confidence_level: UnitFraction
    cvar_alpha: UnitFraction
    seed_statistics: Annotated[tuple[SeedStatistics, ...], Field(min_length=1)]
    methods: Annotated[tuple[MethodStatistics, ...], Field(min_length=1)]
    comparisons: Annotated[tuple[PairwiseComparison, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_identities(self) -> Self:
        seed_keys = {
            (item.method_id, item.optimizer_seed) for item in self.seed_statistics
        }
        method_ids = {item.method_id for item in self.methods}
        comparison_keys = {
            tuple(sorted((item.method_a, item.method_b)))
            for item in self.comparisons
        }
        if (
            len(seed_keys) != len(self.seed_statistics)
            or len(method_ids) != len(self.methods)
            or len(comparison_keys) != len(self.comparisons)
        ):
            raise DuplicateStatisticsEntryError
        return self


class StatisticsReceipt(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["statistics_receipt"]
    manifest_sha256: Sha256
    statistics_sha256: Sha256
    statistics_implementation_sha256: Sha256
