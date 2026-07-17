"""All-pairs seed-level tests and paired hierarchical intervals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from .statistics_core import benjamini_hochberg, cliffs_delta, wilcoxon_p_value
from .statistics_intervals import interval
from .statistics_models import EstimateInterval, MetricValues, PairwiseComparison


@dataclass(frozen=True, slots=True)
class ComparisonRequest:
    method_ids: tuple[str, ...]
    optimizer_seeds: tuple[int, ...]
    seed_metrics: tuple[tuple[MetricValues, ...], ...]
    hierarchical: NDArray[np.float64]
    confidence_level: float


def pairwise_comparisons(
    request: ComparisonRequest,
) -> tuple[PairwiseComparison, ...]:
    pending: list[tuple[int, int, EstimateInterval, float, float]] = []
    for first_index, second_index in combinations(range(len(request.method_ids)), 2):
        first = np.asarray(
            tuple(item.mean for item in request.seed_metrics[first_index]),
            dtype=np.float64,
        )
        second = np.asarray(
            tuple(item.mean for item in request.seed_metrics[second_index]),
            dtype=np.float64,
        )
        differences = first - second
        bootstrap = (
            request.hierarchical[first_index : first_index + 1].reshape(-1)
            - request.hierarchical[second_index : second_index + 1].reshape(-1)
        )
        pending.append(
            (
                first_index,
                second_index,
                interval(differences, bootstrap, request.confidence_level),
                wilcoxon_p_value(differences),
                cliffs_delta(first, second),
            ),
        )
    adjusted = benjamini_hochberg(tuple(item[3] for item in pending))
    return tuple(
        PairwiseComparison(
            method_a=request.method_ids[item[0]],
            method_b=request.method_ids[item[1]],
            seed_pairs=len(request.optimizer_seeds),
            mean_difference=item[2],
            wilcoxon_p=item[3],
            bh_q=q_value,
            cliffs_delta=item[4],
        )
        for item, q_value in zip(pending, adjusted, strict=True)
    )
