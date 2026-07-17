from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.figure_models import FigureReceipt, FigureSidecar
from quantum_battery_rl.benchmark.figure_runner import (
    FigureRequest,
    StaleFigureError,
    run_figures,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
CANONICAL_ROOT: Final = PROJECT_ROOT / "results" / "canonical"


def test_figures_are_complete_traceable_and_deterministic(tmp_path: Path) -> None:
    # Given
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    # When
    first = run_figures(FigureRequest(PROJECT_ROOT, CANONICAL_ROOT, first_root))
    resumed = run_figures(FigureRequest(PROJECT_ROOT, CANONICAL_ROOT, first_root))
    independent = run_figures(FigureRequest(PROJECT_ROOT, CANONICAL_ROOT, second_root))

    # Then
    assert not first.reused
    assert resumed.reused
    assert not independent.reused
    assert first.output_sha256 == independent.output_sha256
    assert first.figure_count == 4
    receipt = FigureReceipt.model_validate_json(
        (first_root / "figure_receipt.json").read_bytes(),
    )
    assert set(receipt.output_sha256) == {
        f"{figure_id}.{suffix}"
        for figure_id in (
            "uncertainty_coverage",
            "convergence_cost",
            "heldout_distributions",
            "mean_cvar_comparison",
        )
        for suffix in ("json", "pdf", "png")
    }
    heldout = FigureSidecar.model_validate_json(
        (first_root / "heldout_distributions.json").read_bytes(),
    )
    assert sum(len(series.y) for series in heldout.series) == 53_760
    assert len(heldout.series) == 7


def test_figures_reject_tampered_render(tmp_path: Path) -> None:
    # Given
    output_root = tmp_path / "figures"
    request = FigureRequest(PROJECT_ROOT, CANONICAL_ROOT, output_root)
    _ = run_figures(request)
    _ = (output_root / "mean_cvar_comparison.png").write_bytes(b"tampered")

    # When / Then
    with pytest.raises(StaleFigureError):
        _ = run_figures(request)
