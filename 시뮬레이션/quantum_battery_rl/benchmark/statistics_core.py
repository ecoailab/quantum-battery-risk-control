"""Pure seed-level statistics and paired resampling primitives."""

from __future__ import annotations

from math import ceil

import numpy as np
from numpy.typing import NDArray

from .statistics_models import MetricValues


class EmptyStatisticsInputError(ValueError):
    """Raised when a statistical estimand receives no observations."""


class InvalidBootstrapShapeError(ValueError):
    """Raised when hierarchical bootstrap input is not method/seed/draw shaped."""


def compute_metric_values(
    values: NDArray[np.float64],
    alpha: float,
) -> MetricValues:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if ordered.size == 0:
        raise EmptyStatisticsInputError
    tail_count = max(1, ceil(alpha * ordered.size))
    return MetricValues(
        mean=float(np.mean(ordered)),
        median=float(np.median(ordered)),
        percentile_10=float(np.quantile(ordered, 0.1)),
        cvar_0_1=float(np.mean(ordered[:tail_count])),
        minimum=float(ordered[0:1].reshape(())),
    )


def paired_two_level_bootstrap(
    values: NDArray[np.float64],
    samples: int,
    seed: int,
) -> NDArray[np.float64]:
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 3:
        raise InvalidBootstrapShapeError
    method_count = int(np.size(observations, axis=0))
    seed_count = int(np.size(observations, axis=1))
    draw_count = int(np.size(observations, axis=2))
    random = np.random.default_rng(seed)
    result = np.empty((method_count, samples), dtype=np.float64)
    seed_indices: NDArray[np.int64] = random.integers(
        0,
        seed_count,
        size=(samples, seed_count),
    )
    draw_indices: NDArray[np.int64] = random.integers(
        0,
        draw_count,
        size=(samples, seed_count, draw_count),
    )
    for sample_index in range(samples):
        sample_sum = np.zeros(method_count, dtype=np.float64)
        for seed_slot in range(seed_count):
            selected_seed = int(
                seed_indices[
                    sample_index : sample_index + 1,
                    seed_slot : seed_slot + 1,
                ].reshape(()),
            )
            selected_draws = draw_indices[
                sample_index : sample_index + 1,
                seed_slot : seed_slot + 1,
                :,
            ].reshape(-1)
            seed_values = observations[
                :,
                selected_seed : selected_seed + 1,
                :,
            ].reshape((method_count, draw_count))
            sampled_values: NDArray[np.float64] = np.take(
                seed_values,
                selected_draws,
                axis=1,
            )
            sample_sum += np.mean(sampled_values, axis=1)
        result[:, sample_index] = sample_sum / seed_count
    result.setflags(write=False)
    return result


def cliffs_delta(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1, 1)
    right = np.asarray(second, dtype=np.float64).reshape(1, -1)
    if left.size == 0 or right.size == 0:
        raise EmptyStatisticsInputError
    greater = np.count_nonzero(left > right)
    lower = np.count_nonzero(left < right)
    return float((greater - lower) / (left.size * right.size))


def wilcoxon_p_value(differences: NDArray[np.float64]) -> float:
    nonzero = np.asarray(differences, dtype=np.float64).reshape(-1)
    nonzero = nonzero[nonzero != 0.0]
    if nonzero.size == 0:
        return 1.0
    ranks = _average_ranks(np.abs(nonzero))
    observed = float(np.sum(ranks[nonzero > 0.0]))
    center = float(np.sum(ranks) / 2.0)
    observed_distance = abs(observed - center)
    extreme = 0
    permutation_count = 1 << nonzero.size
    for mask in range(permutation_count):
        positive_sum = sum(
            float(ranks[index : index + 1].reshape(()))
            for index in range(nonzero.size)
            if mask & (1 << index)
        )
        extreme += int(abs(positive_sum - center) >= observed_distance - 1e-12)
    return extreme / permutation_count


def benjamini_hochberg(p_values: tuple[float, ...]) -> tuple[float, ...]:
    if not p_values:
        raise EmptyStatisticsInputError
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        original_index, value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, value * len(ordered) / rank)
        adjusted[original_index] = min(1.0, running)
    return tuple(adjusted)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        ordered_value = float(values[order[start] : order[start] + 1].reshape(()))
        while stop < values.size:
            candidate = float(values[order[stop] : order[stop] + 1].reshape(()))
            if candidate != ordered_value:
                break
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks
