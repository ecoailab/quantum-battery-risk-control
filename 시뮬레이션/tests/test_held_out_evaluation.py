from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_models import HeldOutResultsArtifact
from quantum_battery_rl.benchmark.artifact_store import (
    ArtifactReadRequest,
    ArtifactStore,
    StaleArtifactError,
)
from quantum_battery_rl.benchmark.held_out_evaluation import (
    HeldOutEvaluationRequest,
    run_held_out_evaluation,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
MANIFEST_PATH: Final = PROJECT_ROOT / "canonical_manifest.json"
CANONICAL_ROOT: Final = PROJECT_ROOT / "results" / "canonical"


def test_held_out_evaluation_is_complete_paired_and_resume_safe(
    tmp_path: Path,
) -> None:
    # Given
    output_root = tmp_path / "held-out"
    request = HeldOutEvaluationRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        CANONICAL_ROOT,
        output_root,
    )

    # When
    first = run_held_out_evaluation(request)
    second = run_held_out_evaluation(request)

    # Then
    assert first.record_count == 70 * 768
    assert not first.reused
    assert second.reused
    assert second.raw_results_sha256 == first.raw_results_sha256
    raw = ArtifactStore(output_root).read(
        HeldOutResultsArtifact,
        ArtifactReadRequest(Path("raw_results.json"), first.raw_results_sha256),
    )
    assert raw.evaluation_split == "test"
    assert len(raw.fit_artifact_sha256s) == 70
    assert len(raw.records) == 70 * 768
    assert len({record.method_id for record in raw.records}) == 7
    assert {record.optimizer_seed for record in raw.records} == set(range(42, 52))
    assert {record.severity_fraction for record in raw.records} == {0.1, 0.3, 0.5}
    assert all(record.draw_id.startswith("test-") for record in raw.records)
    assert all(record.runtime_seconds >= 0.0 for record in raw.records)

    changed_value = 0.0 if raw.records[0].final_ergotropy > 0.0 else 1.0
    changed_record = raw.records[0].model_copy(
        update={"final_ergotropy": changed_value},
    )
    changed_raw = raw.model_copy(update={"records": (changed_record, *raw.records[1:])})
    _ = ArtifactStore(output_root).write(Path("raw_results.json"), changed_raw)
    with pytest.raises(StaleArtifactError):
        _ = run_held_out_evaluation(request)
