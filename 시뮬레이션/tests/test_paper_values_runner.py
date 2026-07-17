from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_models import PaperValuesArtifact
from quantum_battery_rl.benchmark.artifact_store import (
    ArtifactReadRequest,
    ArtifactStore,
)
from quantum_battery_rl.benchmark.paper_values_runner import (
    PaperValuesReceipt,
    PaperValuesRequest,
    StalePaperValuesError,
    run_paper_values,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
MANIFEST_PATH: Final = PROJECT_ROOT / "canonical_manifest.json"
CANONICAL_ROOT: Final = PROJECT_ROOT / "results" / "canonical"


def test_paper_values_are_canonical_complete_and_resume_safe(tmp_path: Path) -> None:
    # Given
    output_root = tmp_path / "canonical"
    generated_root = tmp_path / "generated"
    request = PaperValuesRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        CANONICAL_ROOT,
        output_root,
        generated_root,
    )

    # When
    first = run_paper_values(request)
    second = run_paper_values(request)

    # Then
    assert not first.reused
    assert second.reused
    assert second.paper_values_sha256 == first.paper_values_sha256
    receipt = PaperValuesReceipt.model_validate_json(
        (output_root / "paper_values_receipt.json").read_bytes(),
    )
    artifact = ArtifactStore(output_root).read(
        PaperValuesArtifact,
        ArtifactReadRequest(Path("paper_values.json"), receipt.paper_values_sha256),
    )
    values = {item.key: item for item in artifact.values}
    assert values["method.cvar.mean.estimate"].value == pytest.approx(0.944274)
    assert values["method.ppo.mean.estimate"].value == pytest.approx(
        0.0085913,
        abs=1e-7,
    )
    assert values["claim.ppo_superiority_all"].value == 0.0
    assert sum(key.endswith(".mean.estimate") for key in values) == 28
    forbidden = ("ceiling", "ionq", "platform", "two_qubit")
    assert not any(token in key for key in values for token in forbidden)
    assert "0.950" not in (output_root / "paper_values.json").read_text(
        encoding="utf-8",
    )
    assert "0.950" not in (generated_root / "results_macros.tex").read_text(
        encoding="utf-8",
    )
    assert set(receipt.generated_sha256) == {
        "comparison_table.tex",
        "cost_table.tex",
        "method_summary_table.tex",
        "results_macros.tex",
        "severity_table.tex",
    }


def test_paper_values_reject_tampered_generated_fragment(tmp_path: Path) -> None:
    # Given
    output_root = tmp_path / "canonical"
    generated_root = tmp_path / "generated"
    request = PaperValuesRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        CANONICAL_ROOT,
        output_root,
        generated_root,
    )
    _ = run_paper_values(request)
    _ = (generated_root / "cost_table.tex").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(StalePaperValuesError):
        _ = run_paper_values(request)
