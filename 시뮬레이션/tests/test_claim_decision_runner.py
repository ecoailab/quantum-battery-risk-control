from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_models import ClaimDecisionArtifact
from quantum_battery_rl.benchmark.artifact_store import (
    ArtifactReadRequest,
    ArtifactStore,
    StaleArtifactError,
)
from quantum_battery_rl.benchmark.claim_decision_runner import (
    ClaimDecisionReceipt,
    ClaimDecisionRequest,
    run_claim_decision,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
MANIFEST_PATH: Final = PROJECT_ROOT / "canonical_manifest.json"
CANONICAL_ROOT: Final = PROJECT_ROOT / "results" / "canonical"


def test_claim_decision_is_direction_aware_neutral_and_resume_safe(
    tmp_path: Path,
) -> None:
    # Given
    output_root = tmp_path / "claim"
    request = ClaimDecisionRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        CANONICAL_ROOT,
        output_root,
    )

    # When
    first = run_claim_decision(request)
    second = run_claim_decision(request)

    # Then
    assert not first.reused
    assert second.reused
    assert second.claim_decision_sha256 == first.claim_decision_sha256
    receipt = ClaimDecisionReceipt.model_validate_json(
        (output_root / "claim_decision_receipt.json").read_bytes(),
    )
    decision = ArtifactStore(output_root).read(
        ClaimDecisionArtifact,
        ArtifactReadRequest(
            Path("claim_decision.json"),
            receipt.claim_decision_sha256,
        ),
    )
    gates = {gate.gate_id: gate.passed for gate in decision.gates}
    assert decision.branch == "neutral"
    assert decision.selected_method_id is None
    assert decision.comparator_method_id == "cvar-0.1-lbfgsb-finite-difference"
    assert gates["ppo_q_below_0.05"]
    assert gates["ppo_abs_cliffs_delta_at_least_0.33"]
    assert gates["ppo_mean_ci_excludes_zero"]
    assert not gates["ppo_mean_direction_positive"]
    assert gates["ppo_cvar_same_sign"]
    assert not gates["ppo_cvar_direction_positive"]
    assert not gates["ppo_superiority_all"]

    changed = decision.model_copy(update={"comparator_method_id": "random-seeded-reference"})
    _ = ArtifactStore(output_root).write(Path("claim_decision.json"), changed)
    with pytest.raises(StaleArtifactError):
        _ = run_claim_decision(request)
