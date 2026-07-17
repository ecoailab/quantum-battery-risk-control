"""Direction-aware fixed claim branch from canonical statistics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifact_models import ClaimDecisionArtifact, GateResult
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .manifest import load_manifest
from .smoke_fitters import source_hash
from .statistics_models import (
    PairwiseComparison,
    StatisticsArtifact,
    StatisticsReceipt,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PPO_METHOD = "ppo-time-only-domain-randomized"


class StaleClaimDecisionError(ValueError):
    """Raised when claim inputs or persisted output links are stale."""


class ClaimDecisionReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["1.0"]
    artifact_type: Literal["claim_decision_receipt"]
    manifest_sha256: Sha256
    statistics_sha256: Sha256
    claim_decision_sha256: Sha256
    claim_implementation_sha256: Sha256


@dataclass(frozen=True, slots=True)
class ClaimDecisionRequest:
    project_root: Path
    manifest_path: Path
    canonical_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class ClaimDecisionReport:
    claim_decision_sha256: str
    receipt_sha256: str
    branch: str
    reused: bool


@dataclass(frozen=True, slots=True)
class _OrientedComparison:
    estimate: float
    lower: float
    upper: float
    q_value: float
    cliffs_delta: float


def run_claim_decision(request: ClaimDecisionRequest) -> ClaimDecisionReport:
    manifest_sha256 = sha256(request.manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(request.manifest_path)
    statistics_receipt_path = request.canonical_root / "statistics_receipt.json"
    statistics_receipt_payload = statistics_receipt_path.read_bytes()
    statistics_receipt = StatisticsReceipt.model_validate_json(
        statistics_receipt_payload,
    )
    statistics = ArtifactStore(request.canonical_root).read(
        StatisticsArtifact,
        ArtifactReadRequest(
            Path("statistics.json"),
            statistics_receipt.statistics_sha256,
        ),
    )
    implementation_hash = source_hash(
        request.project_root,
        (
            Path("quantum_battery_rl/benchmark/claim_decision_runner.py"),
            Path("quantum_battery_rl/benchmark/artifact_models.py"),
        ),
    )
    existing = _resume_existing(
        request,
        manifest_sha256,
        statistics_receipt.statistics_sha256,
        implementation_hash,
    )
    if existing is not None:
        return existing
    methods = {method.method_id: method for method in statistics.methods}
    if PPO_METHOD not in methods:
        raise StaleClaimDecisionError("statistics do not contain PPO")
    comparator = max(
        (method for method in statistics.methods if method.method_id != PPO_METHOD),
        key=lambda method: method.metrics.mean.estimate,
    )
    comparison = _find_comparison(statistics.comparisons, PPO_METHOD, comparator.method_id)
    oriented = _orient_comparison(comparison, PPO_METHOD)
    ppo = methods[PPO_METHOD]
    cvar_difference = ppo.metrics.cvar_0_1.estimate - comparator.metrics.cvar_0_1.estimate
    q_passed = oriented.q_value < manifest.statistics.q_threshold
    effect_passed = abs(oriented.cliffs_delta) >= manifest.statistics.cliff_threshold
    ci_excludes_zero = oriented.lower > 0.0 or oriented.upper < 0.0
    mean_positive = oriented.estimate > 0.0
    cvar_same_sign = oriented.estimate * cvar_difference > 0.0
    cvar_positive = cvar_difference > 0.0
    superiority = all(
        (
            q_passed,
            effect_passed,
            ci_excludes_zero,
            mean_positive,
            cvar_same_sign,
            cvar_positive,
        ),
    )
    gates = (
        GateResult(
            gate_id="ppo_q_below_0.05",
            passed=q_passed,
            detail=f"oriented q={oriented.q_value:.12g}",
        ),
        GateResult(
            gate_id="ppo_abs_cliffs_delta_at_least_0.33",
            passed=effect_passed,
            detail=f"oriented Cliff's delta={oriented.cliffs_delta:.12g}",
        ),
        GateResult(
            gate_id="ppo_mean_ci_excludes_zero",
            passed=ci_excludes_zero,
            detail=f"oriented CI=[{oriented.lower:.12g}, {oriented.upper:.12g}]",
        ),
        GateResult(
            gate_id="ppo_mean_direction_positive",
            passed=mean_positive,
            detail=f"PPO-minus-comparator mean={oriented.estimate:.12g}",
        ),
        GateResult(
            gate_id="ppo_cvar_same_sign",
            passed=cvar_same_sign,
            detail=f"PPO-minus-comparator CVaR={cvar_difference:.12g}",
        ),
        GateResult(
            gate_id="ppo_cvar_direction_positive",
            passed=cvar_positive,
            detail=f"PPO-minus-comparator CVaR={cvar_difference:.12g}",
        ),
        GateResult(
            gate_id="ppo_superiority_all",
            passed=superiority,
            detail="All predeclared significance, effect, interval, and direction gates.",
        ),
    )
    branch: Literal["method-specific", "neutral"] = (
        "method-specific" if superiority else "neutral"
    )
    selected_method = PPO_METHOD if superiority else None
    rationale = (
        "PPO passed every predeclared superiority gate."
        if superiority
        else "PPO did not pass all predeclared superiority and direction gates."
    )
    decision = ClaimDecisionArtifact(
        schema_version="1.0",
        artifact_type="claim_decision",
        manifest_sha256=manifest_sha256,
        summary_sha256=statistics_receipt.statistics_sha256,
        branch=branch,
        selected_method_id=selected_method,
        comparator_method_id=comparator.method_id,
        rationale=rationale,
        gates=gates,
    )
    store = ArtifactStore(request.output_root)
    decision_receipt = store.write(Path("claim_decision.json"), decision)
    receipt = ClaimDecisionReceipt(
        schema_version="1.0",
        artifact_type="claim_decision_receipt",
        manifest_sha256=manifest_sha256,
        statistics_sha256=statistics_receipt.statistics_sha256,
        claim_decision_sha256=decision_receipt.sha256,
        claim_implementation_sha256=implementation_hash,
    )
    receipt_receipt = store.write(Path("claim_decision_receipt.json"), receipt)
    return ClaimDecisionReport(
        decision_receipt.sha256,
        receipt_receipt.sha256,
        branch,
        False,
    )


def _find_comparison(
    comparisons: tuple[PairwiseComparison, ...],
    first: str,
    second: str,
) -> PairwiseComparison:
    for comparison in comparisons:
        if {comparison.method_a, comparison.method_b} == {first, second}:
            return comparison
    raise StaleClaimDecisionError("best-comparator pair is missing")


def _orient_comparison(
    comparison: PairwiseComparison,
    first: str,
) -> _OrientedComparison:
    if comparison.method_a == first:
        return _OrientedComparison(
            comparison.mean_difference.estimate,
            comparison.mean_difference.lower,
            comparison.mean_difference.upper,
            comparison.bh_q,
            comparison.cliffs_delta,
        )
    return _OrientedComparison(
        -comparison.mean_difference.estimate,
        -comparison.mean_difference.upper,
        -comparison.mean_difference.lower,
        comparison.bh_q,
        -comparison.cliffs_delta,
    )


def _resume_existing(
    request: ClaimDecisionRequest,
    manifest_hash: str,
    statistics_hash: str,
    implementation_hash: str,
) -> ClaimDecisionReport | None:
    decision_path = request.output_root / "claim_decision.json"
    receipt_path = request.output_root / "claim_decision_receipt.json"
    if not decision_path.is_file() and not receipt_path.is_file():
        return None
    if not decision_path.is_file() or not receipt_path.is_file():
        raise StaleClaimDecisionError("claim decision and receipt must coexist")
    receipt_payload = receipt_path.read_bytes()
    receipt = ClaimDecisionReceipt.model_validate_json(receipt_payload)
    decision = ArtifactStore(request.output_root).read(
        ClaimDecisionArtifact,
        ArtifactReadRequest(
            Path("claim_decision.json"),
            receipt.claim_decision_sha256,
        ),
    )
    if (
        receipt.manifest_sha256 != manifest_hash
        or receipt.statistics_sha256 != statistics_hash
        or receipt.claim_implementation_sha256 != implementation_hash
        or decision.manifest_sha256 != manifest_hash
        or decision.summary_sha256 != statistics_hash
    ):
        raise StaleClaimDecisionError("stale claim decision")
    return ClaimDecisionReport(
        receipt.claim_decision_sha256,
        sha256(receipt_payload).hexdigest(),
        decision.branch,
        True,
    )
