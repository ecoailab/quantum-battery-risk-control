from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_store import (
    ArtifactReadRequest,
    ArtifactStore,
    StaleArtifactError,
)
from quantum_battery_rl.benchmark.statistics_models import (
    StatisticsArtifact,
    StatisticsReceipt,
)
from quantum_battery_rl.benchmark.statistics_runner import (
    StatisticsRequest,
    run_statistics,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
MANIFEST_PATH: Final = PROJECT_ROOT / "canonical_manifest.json"
CANONICAL_ROOT: Final = PROJECT_ROOT / "results" / "canonical"


def test_statistics_runner_is_complete_seed_level_and_resume_safe(
    tmp_path: Path,
) -> None:
    # Given
    output_root = tmp_path / "statistics"
    request = StatisticsRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        CANONICAL_ROOT,
        output_root,
    )

    # When
    first = run_statistics(request)
    second = run_statistics(request)

    # Then
    assert not first.reused
    assert second.reused
    assert second.statistics_sha256 == first.statistics_sha256
    assert first.seed_statistic_count == 70
    assert first.method_count == 7
    assert first.comparison_count == 21
    receipt = StatisticsReceipt.model_validate_json(
        (output_root / "statistics_receipt.json").read_bytes(),
    )
    artifact = ArtifactStore(output_root).read(
        StatisticsArtifact,
        ArtifactReadRequest(Path("statistics.json"), receipt.statistics_sha256),
    )
    assert len(artifact.seed_statistics) == 70
    assert len(artifact.methods) == 7
    assert all(len(method.severity) == 3 for method in artifact.methods)
    assert len(artifact.comparisons) == 21
    assert all(comparison.bh_q >= comparison.wilcoxon_p for comparison in artifact.comparisons)

    changed = artifact.model_copy(update={"bootstrap_seed": artifact.bootstrap_seed + 1})
    _ = ArtifactStore(output_root).write(Path("statistics.json"), changed)
    with pytest.raises(StaleArtifactError):
        _ = run_statistics(request)
