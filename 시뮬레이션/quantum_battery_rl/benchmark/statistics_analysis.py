"""Orchestrate seed-level summaries and all-pairs comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .controller import FitCost
from .statistics_comparisons import ComparisonRequest, pairwise_comparisons
from .statistics_core import compute_metric_values, paired_two_level_bootstrap
from .statistics_intervals import (
    SeverityRequest,
    metric_columns,
    metric_intervals,
    severity_statistics,
    total_cost,
)
from .statistics_models import (
    MethodStatistics,
    MetricValues,
    PairwiseComparison,
    SeedStatistics,
)


class StatisticsCardinalityError(ValueError):
    """Raised when statistical outputs do not cover every declared method."""


@dataclass(frozen=True, slots=True)
class StatisticsInputs:
    method_ids: tuple[str, ...]
    optimizer_seeds: tuple[int, ...]
    severities: NDArray[np.float64]
    values: NDArray[np.float64]
    fit_costs: tuple[FitCost, ...]
    cvar_alpha: float
    bootstrap_samples: int
    confidence_level: float
    bootstrap_seed: int


@dataclass(frozen=True, slots=True)
class StatisticsComponents:
    seed_statistics: tuple[SeedStatistics, ...]
    methods: tuple[MethodStatistics, ...]
    comparisons: tuple[PairwiseComparison, ...]


def analyze_statistics(inputs: StatisticsInputs) -> StatisticsComponents:
    method_count = len(inputs.method_ids)
    seed_count = len(inputs.optimizer_seeds)
    random = np.random.default_rng(inputs.bootstrap_seed)
    seed_resamples: NDArray[np.int64] = random.integers(
        0,
        seed_count,
        size=(inputs.bootstrap_samples, seed_count),
    )
    hierarchical = paired_two_level_bootstrap(
        inputs.values,
        inputs.bootstrap_samples,
        inputs.bootstrap_seed,
    )
    seed_rows: list[SeedStatistics] = []
    method_rows: list[MethodStatistics] = []
    seed_metrics_by_method: list[tuple[MetricValues, ...]] = []
    for method_index, method_id in enumerate(inputs.method_ids):
        method_seed_metrics = tuple(
            compute_metric_values(
                inputs.values[
                    method_index : method_index + 1,
                    seed_index : seed_index + 1,
                    :,
                ].reshape(-1),
                inputs.cvar_alpha,
            )
            for seed_index in range(seed_count)
        )
        seed_metrics_by_method.append(method_seed_metrics)
        seed_rows.extend(
            SeedStatistics(method_id=method_id, optimizer_seed=seed, metrics=metrics)
            for seed, metrics in zip(
                inputs.optimizer_seeds,
                method_seed_metrics,
                strict=True,
            )
        )
        columns = metric_columns(method_seed_metrics)
        mean_bootstrap = hierarchical[
            method_index : method_index + 1,
            :,
        ].reshape(-1)
        method_rows.append(
            MethodStatistics(
                method_id=method_id,
                seed_count=seed_count,
                metrics=metric_intervals(
                    columns,
                    seed_resamples,
                    inputs.confidence_level,
                    mean_bootstrap,
                ),
                severity=severity_statistics(
                    SeverityRequest(
                        inputs.values,
                        inputs.severities,
                        method_index,
                        seed_count,
                        inputs.cvar_alpha,
                        seed_resamples,
                        inputs.confidence_level,
                    ),
                ),
                fit_cost_total=total_cost(
                    inputs.fit_costs[
                        method_index * seed_count : (method_index + 1) * seed_count
                    ],
                ),
            ),
        )
    if len(method_rows) != method_count:
        raise StatisticsCardinalityError
    comparisons_rows = pairwise_comparisons(
        ComparisonRequest(
            inputs.method_ids,
            inputs.optimizer_seeds,
            tuple(seed_metrics_by_method),
            hierarchical,
            inputs.confidence_level,
        ),
    )
    return StatisticsComponents(
        tuple(seed_rows),
        tuple(method_rows),
        comparisons_rows,
    )
