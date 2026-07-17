"""Derive traceable publication values from canonical statistics."""

from __future__ import annotations

from typing import Final

from .artifact_models import ClaimDecisionArtifact, PaperValue
from .statistics_models import EstimateInterval, MetricIntervals, StatisticsArtifact


METHOD_NAMES: Final = (
    ("nominal-lbfgsb-finite-difference", "nominal", "Nominal"),
    ("saa-mean-lbfgsb-finite-difference", "saa", "SAA mean"),
    ("cvar-0.1-lbfgsb-finite-difference", "cvar", "CVaR"),
    ("ppo-time-only-domain-randomized", "ppo", "PPO"),
    ("bang-bang-ensemble-grid", "bang_bang", "Bang--bang"),
    ("sinusoidal-ensemble-grid", "sinusoidal", "Sinusoidal"),
    ("random-seeded-reference", "random", "Random"),
)


def derive_values(
    analysis: tuple[int, float, float],
    statistics_source: tuple[str, StatisticsArtifact],
    claim_source: tuple[str, ClaimDecisionArtifact],
) -> tuple[PaperValue, ...]:
    """Project canonical evidence into sorted, source-addressed display values."""
    bootstrap_samples, confidence_level, cvar_alpha = analysis
    statistics_hash, statistics = statistics_source
    claim_hash, claim = claim_source
    values: list[PaperValue] = []
    _append(
        values,
        "analysis.bootstrap_samples",
        bootstrap_samples,
        ".0f",
        statistics_hash,
        "bootstrap_samples",
    )
    _append(
        values,
        "analysis.confidence_level",
        confidence_level,
        ".2f",
        statistics_hash,
        "confidence_level",
    )
    _append(
        values, "analysis.cvar_alpha", cvar_alpha, ".2f", statistics_hash, "cvar_alpha"
    )
    aliases = {method_id: alias for method_id, alias, _ in METHOD_NAMES}
    for method in statistics.methods:
        alias = aliases[method.method_id]
        _append(
            values,
            f"method.{alias}.seed_count",
            method.seed_count,
            ".0f",
            statistics_hash,
            f"methods[{method.method_id}].seed_count",
        )
        for metric, interval in _metric_intervals(method.metrics):
            _append_interval(
                values,
                f"method.{alias}.{metric}",
                interval,
                statistics_hash,
                f"methods[{method.method_id}].metrics.{metric}",
            )
        for severity in method.severity:
            severity_name = f"s{int(severity.severity_fraction * 100)}"
            for metric, interval in _metric_intervals(severity.metrics):
                _append_interval(
                    values,
                    f"severity.{alias}.{severity_name}.{metric}",
                    interval,
                    statistics_hash,
                    f"methods[{method.method_id}].severity[{severity.severity_fraction}].metrics.{metric}",
                )
        for field, value in (
            ("objective_evaluations", method.fit_cost_total.objective_evaluations),
            ("gradient_evaluations", method.fit_cost_total.gradient_evaluations),
            ("environment_steps", method.fit_cost_total.environment_steps),
            ("wall_time_seconds", method.fit_cost_total.wall_time_seconds),
        ):
            spec = ".2f" if field == "wall_time_seconds" else ".0f"
            _append(
                values,
                f"cost.{alias}.{field}",
                value,
                spec,
                statistics_hash,
                f"methods[{method.method_id}].fit_cost_total.{field}",
            )
    for comparison in statistics.comparisons:
        pair = "_vs_".join((aliases[comparison.method_a], aliases[comparison.method_b]))
        selector = f"comparisons[{comparison.method_a},{comparison.method_b}]"
        _append_interval(
            values,
            f"comparison.{pair}.mean_difference",
            comparison.mean_difference,
            statistics_hash,
            f"{selector}.mean_difference",
        )
        for field, value in (
            ("wilcoxon_p", comparison.wilcoxon_p),
            ("bh_q", comparison.bh_q),
            ("cliffs_delta", comparison.cliffs_delta),
        ):
            _append(
                values,
                f"comparison.{pair}.{field}",
                value,
                ".6g",
                statistics_hash,
                f"{selector}.{field}",
            )
    for gate in claim.gates:
        _append(
            values,
            f"claim.{gate.gate_id}",
            float(gate.passed),
            ".0f",
            claim_hash,
            f"gates[{gate.gate_id}].passed",
        )
    return tuple(sorted(values, key=lambda item: item.key))


def _metric_intervals(
    metrics: MetricIntervals,
) -> tuple[tuple[str, EstimateInterval], ...]:
    return (
        ("mean", metrics.mean),
        ("median", metrics.median),
        ("percentile_10", metrics.percentile_10),
        ("cvar_0_1", metrics.cvar_0_1),
        ("minimum", metrics.minimum),
    )


def _append_interval(
    values: list[PaperValue],
    key: str,
    interval: EstimateInterval,
    source_hash: str,
    selector: str,
) -> None:
    for field, value in (
        ("estimate", interval.estimate),
        ("lower", interval.lower),
        ("upper", interval.upper),
    ):
        _append(
            values, f"{key}.{field}", value, ".6f", source_hash, f"{selector}.{field}"
        )


def _append(
    values: list[PaperValue],
    key: str,
    value: float,
    spec: str,
    source_hash: str,
    selector: str,
) -> None:
    values.append(
        PaperValue(
            key=key,
            value=float(value),
            formatted=format(value, spec),
            source_artifact_sha256=source_hash,
            source_selector=selector,
        )
    )
