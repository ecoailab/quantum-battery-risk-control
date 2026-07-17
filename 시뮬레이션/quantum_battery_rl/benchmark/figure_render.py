"""Accessible, print-oriented Matplotlib rendering for canonical figures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .figure_models import FigureId, FigureSeries, FigureSidecar
from .paper_values_content import METHOD_NAMES


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#4D4D4D")


def render_figure(sidecar: FigureSidecar, output_root: Path) -> tuple[Path, Path]:
    """Render one sidecar to deterministic PNG and PDF files."""
    _configure()
    figure = _renderers()[sidecar.figure_id](sidecar)
    png = output_root / f"{sidecar.figure_id}.png"
    pdf = output_root / f"{sidecar.figure_id}.pdf"
    figure.savefig(
        png, dpi=300, bbox_inches="tight", metadata={"Software": "quantum-battery-rl"}
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "quantum-battery-rl",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return png, pdf


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DDE3",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _render_uncertainty(sidecar: FigureSidecar) -> Figure:
    figure = Figure(figsize=(5.2, 3.5), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    for index, series in enumerate(sidecar.series):
        boundary = _convex_hull(series.x, series.y)
        _ = axis.fill(
            tuple(point[0] for point in boundary),
            tuple(point[1] for point in boundary),
            color=COLORS[index],
            alpha=0.08,
            linewidth=0,
        )
        _ = axis.scatter(
            series.x,
            series.y,
            s=10,
            alpha=0.5,
            color=COLORS[index],
            marker=("o", "s", "^")[index],
            label=series.label,
            linewidths=0,
        )
    _ = axis.set(
        title="Synthetic quasi-static uncertainty coverage",
        xlabel=sidecar.x_label,
        ylabel=sidecar.y_label,
    )
    _ = axis.legend(frameon=False, title="Severity", loc="upper left")
    return figure


def _render_convergence_cost(sidecar: FigureSidecar) -> Figure:
    figure = Figure(figsize=(7.2, 3.2), constrained_layout=True)
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    convergence = tuple(
        series for series in sidecar.series if series.panel == "convergence"
    )
    costs = tuple(series for series in sidecar.series if series.panel == "cost")
    for index, series in enumerate(convergence):
        _ = left.plot(
            series.x,
            series.y,
            marker=("o", "s", "^", "D")[index],
            color=COLORS[index],
            linewidth=1.4,
            label=series.label,
        )
    left.set_xscale("log")
    _ = left.set(
        title="Validation convergence",
        xlabel="Declared budget (log scale)",
        ylabel="Mean ergotropy",
    )
    _ = left.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )
    bars = right.bar(
        tuple(series.x[0] for series in costs),
        tuple(series.y[0] for series in costs),
        color=COLORS,
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, hatch in zip(bars, ("", "//", "xx", "..", "--", "++", "oo"), strict=True):
        bar.set_hatch(hatch)
    _method_ticks(right)
    right.set_yscale("log")
    _ = right.set(title="Aggregate fitting cost", ylabel="Wall time (s, log scale)")
    return figure


def _render_distributions(sidecar: FigureSidecar) -> Figure:
    figure = Figure(figsize=(7.2, 3.5), constrained_layout=True)
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    _distribution_panel(left, sidecar.series[:3], "Optimized controls", (0.91, 0.955))
    _distribution_panel(right, sidecar.series[3:], "Reference controls", (0.0, 0.36))
    left.set_ylabel(sidecar.y_label)
    figure.suptitle("Held-out ergotropy distributions")
    return figure


def _render_mean_cvar(sidecar: FigureSidecar) -> Figure:
    figure = Figure(figsize=(8.4, 3.5), constrained_layout=True)
    left = figure.add_subplot(1, 3, 1)
    center = figure.add_subplot(1, 3, 2)
    right = figure.add_subplot(1, 3, 3)
    _metric_panel(left, sidecar, slice(0, 3), "Optimized controls", (0.91, 0.955))
    _metric_panel(center, sidecar, slice(3, 5), "Low-output references", (0.007, 0.010))
    _metric_panel(right, sidecar, slice(5, 7), "Other references", (0.02, 0.35))
    left.set_ylabel(sidecar.y_label)
    _ = right.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )
    figure.suptitle("Mean and lower-tail performance")
    return figure


def _distribution_panel(
    axis: Axes,
    series_group: tuple[FigureSeries, ...],
    title: str,
    limits: tuple[float, float],
) -> None:
    _ = axis.violinplot(
        tuple(series.y for series in series_group),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    labels = tuple(series.label.replace("--", "-") for series in series_group)
    _ = axis.set_xticks(
        tuple(range(1, len(labels) + 1)), labels, rotation=25, ha="right"
    )
    axis.set_ylim(limits)
    axis.set_title(title)


def _metric_panel(
    axis: Axes,
    sidecar: FigureSidecar,
    selection: slice,
    title: str,
    limits: tuple[float, float],
) -> None:
    offsets = (-0.13, 0.13)
    base = np.arange(len(METHOD_NAMES))[selection]
    for method_index in range(len(base)):
        _ = axis.plot(
            (base[method_index] + offsets[0], base[method_index] + offsets[1]),
            tuple(series.y[selection][method_index] for series in sidecar.series),
            color="#A7ADB5",
            linewidth=0.8,
            zorder=1,
        )
    for index, series in enumerate(sidecar.series):
        y = np.asarray(series.y[selection])
        lower = y - np.asarray(series.lower[selection])
        upper = np.asarray(series.upper[selection]) - y
        _ = axis.errorbar(
            base + offsets[index],
            y,
            yerr=np.vstack((lower, upper)),
            marker=("o", "s")[index],
            linestyle="none",
            capsize=3,
            color=COLORS[index],
            label=series.label,
            zorder=2,
        )
    labels = tuple(label.replace("--", "-") for _, _, label in METHOD_NAMES[selection])
    _ = axis.set_xticks(base, labels, rotation=25, ha="right")
    axis.set_ylim(limits)
    axis.set_title(title)


def _method_ticks(axis: Axes, start: int = 0) -> None:
    positions = tuple(range(start, start + len(METHOD_NAMES)))
    labels = ("Nominal", "SAA", "CVaR", "PPO", "Bang-\nbang", "Sine", "Random")
    axis.set_xticks(
        positions,
        labels,
        rotation=20,
        ha="right",
    )


def _renderers() -> dict[FigureId, Callable[[FigureSidecar], Figure]]:
    return {
        "uncertainty_coverage": _render_uncertainty,
        "convergence_cost": _render_convergence_cost,
        "heldout_distributions": _render_distributions,
        "mean_cvar_comparison": _render_mean_cvar,
    }


def _convex_hull(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    points = sorted(set(zip(x_values, y_values, strict=True)))

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            _ = lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            _ = upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])
