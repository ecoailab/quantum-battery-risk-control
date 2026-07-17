"""Confidence intervals, severity summaries, and fit-cost aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .controller import FitCost
from .statistics_core import compute_metric_values
from .statistics_models import (
    EstimateInterval,
    MetricIntervals,
    MetricValues,
    SeverityStatistics,
)


@dataclass(frozen=True, slots=True)
class MetricColumns:
    mean: NDArray[np.float64]
    median: NDArray[np.float64]
    percentile_10: NDArray[np.float64]
    cvar_0_1: NDArray[np.float64]
    minimum: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SeverityRequest:
    values: NDArray[np.float64]
    severities: NDArray[np.float64]
    method_index: int
    seed_count: int
    cvar_alpha: float
    seed_resamples: NDArray[np.int64]
    confidence_level: float


def metric_columns(metrics: tuple[MetricValues, ...]) -> MetricColumns:
    return MetricColumns(
        np.asarray(tuple(item.mean for item in metrics), dtype=np.float64),
        np.asarray(tuple(item.median for item in metrics), dtype=np.float64),
        np.asarray(tuple(item.percentile_10 for item in metrics), dtype=np.float64),
        np.asarray(tuple(item.cvar_0_1 for item in metrics), dtype=np.float64),
        np.asarray(tuple(item.minimum for item in metrics), dtype=np.float64),
    )


def metric_intervals(
    columns: MetricColumns,
    seed_resamples: NDArray[np.int64],
    confidence_level: float,
    mean_bootstrap: NDArray[np.float64] | None = None,
) -> MetricIntervals:
    mean_samples = (
        seed_bootstrap(columns.mean, seed_resamples)
        if mean_bootstrap is None
        else mean_bootstrap
    )
    return MetricIntervals(
        mean=interval(columns.mean, mean_samples, confidence_level),
        median=interval(
            columns.median,
            seed_bootstrap(columns.median, seed_resamples),
            confidence_level,
        ),
        percentile_10=interval(
            columns.percentile_10,
            seed_bootstrap(columns.percentile_10, seed_resamples),
            confidence_level,
        ),
        cvar_0_1=interval(
            columns.cvar_0_1,
            seed_bootstrap(columns.cvar_0_1, seed_resamples),
            confidence_level,
        ),
        minimum=interval(
            columns.minimum,
            seed_bootstrap(columns.minimum, seed_resamples),
            confidence_level,
        ),
    )


def seed_bootstrap(
    values: NDArray[np.float64],
    seed_resamples: NDArray[np.int64],
) -> NDArray[np.float64]:
    sampled: NDArray[np.float64] = np.take(values, seed_resamples)
    return np.mean(sampled, axis=1)


def interval(
    observed: NDArray[np.float64],
    bootstrap: NDArray[np.float64],
    confidence_level: float,
) -> EstimateInterval:
    estimate = float(np.mean(observed))
    tail = (1.0 - confidence_level) / 2.0
    lower = min(estimate, float(np.quantile(bootstrap, tail)))
    upper = max(estimate, float(np.quantile(bootstrap, 1.0 - tail)))
    return EstimateInterval(estimate=estimate, lower=lower, upper=upper)


def severity_statistics(request: SeverityRequest) -> tuple[SeverityStatistics, ...]:
    rows: list[SeverityStatistics] = []
    unique: NDArray[np.float64] = np.unique(request.severities)
    for severity_index in range(int(np.size(unique))):
        severity = float(unique[severity_index : severity_index + 1].reshape(()))
        mask: NDArray[np.bool_] = np.equal(request.severities, severity)
        draw_indices: NDArray[np.intp] = np.flatnonzero(mask)
        metrics = tuple(
            compute_metric_values(
                np.take(
                    request.values[
                        request.method_index : request.method_index + 1,
                        seed_index : seed_index + 1,
                        :,
                    ].reshape(-1),
                    draw_indices,
                ),
                request.cvar_alpha,
            )
            for seed_index in range(request.seed_count)
        )
        rows.append(
            SeverityStatistics(
                severity_fraction=severity,
                metrics=metric_intervals(
                    metric_columns(metrics),
                    request.seed_resamples,
                    request.confidence_level,
                ),
            ),
        )
    return tuple(rows)


def total_cost(costs: tuple[FitCost, ...]) -> FitCost:
    return FitCost(
        objective_evaluations=sum(cost.objective_evaluations for cost in costs),
        gradient_evaluations=sum(cost.gradient_evaluations for cost in costs),
        environment_steps=sum(cost.environment_steps for cost in costs),
        wall_time_seconds=sum(cost.wall_time_seconds for cost in costs),
    )
