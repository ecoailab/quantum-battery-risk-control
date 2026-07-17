from __future__ import annotations

import numpy as np

from quantum_battery_rl.benchmark.statistics_core import (
    benjamini_hochberg,
    cliffs_delta,
    compute_metric_values,
    paired_two_level_bootstrap,
    wilcoxon_p_value,
)


def test_metric_values_use_finite_lower_tail_and_held_out_minimum() -> None:
    # Given
    values = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # When
    metrics = compute_metric_values(values, alpha=0.1)

    # Then
    assert metrics.mean == 0.5
    assert metrics.median == 0.5
    assert metrics.percentile_10 == 0.1
    assert metrics.cvar_0_1 == 0.05
    assert metrics.minimum == 0.0


def test_two_level_bootstrap_preserves_paired_constant_difference() -> None:
    # Given
    baseline = np.arange(24, dtype=np.float64).reshape(3, 8) / 100.0
    values = np.stack((baseline + 0.2, baseline))

    # When
    bootstrapped = paired_two_level_bootstrap(values, samples=200, seed=4404)

    # Then
    assert bootstrapped.shape == (2, 200)
    first = bootstrapped[0:1].reshape(-1)
    second = bootstrapped[1:2].reshape(-1)
    np.testing.assert_allclose(first - second, 0.2, atol=1e-12)


def test_pairwise_effects_and_bh_correction_cover_edge_cases() -> None:
    # Given / When
    positive_delta = cliffs_delta(
        np.asarray([0.8, 0.9, 1.0]),
        np.asarray([0.1, 0.2, 0.3]),
    )
    tied_p = wilcoxon_p_value(np.zeros(10, dtype=np.float64))
    adjusted = benjamini_hochberg((0.01, 0.04, 0.03))

    # Then
    assert positive_delta == 1.0
    assert tied_p == 1.0
    np.testing.assert_allclose(adjusted, (0.03, 0.04, 0.04))
