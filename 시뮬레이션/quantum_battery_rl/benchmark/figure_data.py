"""Canonical data projections for publication figure sidecars."""

from __future__ import annotations

from dataclasses import dataclass

from .artifact_models import HeldOutResultsArtifact, PaperValuesArtifact
from .budget_models import BudgetAmendmentArtifact
from .figure_models import FigureSeries, FigureSidecar
from .paper_values_content import METHOD_NAMES
from .uncertainty import DrawSet


@dataclass(frozen=True, slots=True)
class FigureInputs:
    draws_sha256: str
    draws: DrawSet
    pilot_sha256: str
    pilot: BudgetAmendmentArtifact
    raw_sha256: str
    raw: HeldOutResultsArtifact
    paper_values_sha256: str
    paper_values: PaperValuesArtifact


def build_sidecars(inputs: FigureInputs) -> tuple[FigureSidecar, ...]:
    """Build the four declared figure sidecars from typed canonical inputs."""
    return (
        _uncertainty(inputs),
        _convergence_cost(inputs),
        _heldout(inputs),
        _mean_cvar(inputs),
    )


def _uncertainty(inputs: FigureInputs) -> FigureSidecar:
    series = tuple(
        FigureSeries(
            series_id=f"severity_{severity:.1f}",
            label=f"{severity:.0%}",
            panel="coverage",
            x=tuple(
                draw.t1
                for draw in inputs.draws.draws
                if draw.severity_fraction == severity
            ),
            y=tuple(
                draw.t2
                for draw in inputs.draws.draws
                if draw.severity_fraction == severity
            ),
        )
        for severity in (0.1, 0.3, 0.5)
    )
    return FigureSidecar(
        schema_version="1.0",
        artifact_type="figure_sidecar",
        figure_id="uncertainty_coverage",
        source_artifact_sha256s=(inputs.draws_sha256,),
        x_label="T1 (microseconds)",
        y_label="T2 (microseconds)",
        series=series,
    )


def _convergence_cost(inputs: FigureInputs) -> FigureSidecar:
    labels = {method_id: (alias, label) for method_id, alias, label in METHOD_NAMES}
    convergence = tuple(
        FigureSeries(
            series_id=f"convergence_{labels[decision.method_id][0]}",
            label=labels[decision.method_id][1],
            panel="convergence",
            x=tuple(
                float(stage.requested_budget)
                for stage in inputs.pilot.stages
                if stage.method_id == decision.method_id
            ),
            y=tuple(
                stage.validation_mean
                for stage in inputs.pilot.stages
                if stage.method_id == decision.method_id
            ),
        )
        for decision in inputs.pilot.decisions
    )
    values = {item.key: item.value for item in inputs.paper_values.values}
    cost = tuple(
        FigureSeries(
            series_id=f"cost_{alias}",
            label=label,
            panel="cost",
            x=(float(index),),
            y=(values[f"cost.{alias}.wall_time_seconds"],),
        )
        for index, (_, alias, label) in enumerate(METHOD_NAMES)
    )
    return FigureSidecar(
        schema_version="1.0",
        artifact_type="figure_sidecar",
        figure_id="convergence_cost",
        source_artifact_sha256s=(inputs.pilot_sha256, inputs.paper_values_sha256),
        x_label="Declared budget / method",
        y_label="Validation ergotropy / fit time (s)",
        series=convergence + cost,
    )


def _heldout(inputs: FigureInputs) -> FigureSidecar:
    series = tuple(
        FigureSeries(
            series_id=alias,
            label=label,
            panel="distribution",
            x=(),
            y=tuple(
                record.final_ergotropy
                for record in inputs.raw.records
                if record.method_id == method_id
            ),
        )
        for method_id, alias, label in METHOD_NAMES
    )
    return FigureSidecar(
        schema_version="1.0",
        artifact_type="figure_sidecar",
        figure_id="heldout_distributions",
        source_artifact_sha256s=(inputs.raw_sha256,),
        x_label="Control method",
        y_label="Held-out final ergotropy",
        series=series,
    )


def _mean_cvar(inputs: FigureInputs) -> FigureSidecar:
    values = {item.key: item.value for item in inputs.paper_values.values}
    indexes = tuple(float(index) for index in range(len(METHOD_NAMES)))
    series = tuple(
        FigureSeries(
            series_id=metric,
            label=label,
            panel="comparison",
            x=indexes,
            y=tuple(
                values[f"method.{alias}.{metric}.estimate"]
                for _, alias, _ in METHOD_NAMES
            ),
            lower=tuple(
                values[f"method.{alias}.{metric}.lower"] for _, alias, _ in METHOD_NAMES
            ),
            upper=tuple(
                values[f"method.{alias}.{metric}.upper"] for _, alias, _ in METHOD_NAMES
            ),
        )
        for metric, label in (("mean", "Mean"), ("cvar_0_1", "CVaR 0.1"))
    )
    return FigureSidecar(
        schema_version="1.0",
        artifact_type="figure_sidecar",
        figure_id="mean_cvar_comparison",
        source_artifact_sha256s=(inputs.paper_values_sha256,),
        x_label="Control method",
        y_label="Held-out final ergotropy",
        series=series,
    )
