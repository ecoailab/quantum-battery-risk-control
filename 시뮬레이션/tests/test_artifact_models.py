from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from quantum_battery_rl.benchmark.artifact_models import (
    ClaimDecisionArtifact,
    EvaluationRecord,
    FitArtifact,
    GateResult,
    MethodSummary,
    PaperValue,
    PaperValuesArtifact,
    RawResultsArtifact,
    SummaryArtifact,
)
from quantum_battery_rl.benchmark.controller import (
    FitCost,
    FitMetadata,
    FitProvenance,
    FittedController,
    ObjectiveContribution,
)


def _hash(character: str) -> str:
    return character * 64


def _cost() -> FitCost:
    return FitCost(
        objective_evaluations=10,
        gradient_evaluations=2,
        environment_steps=1_000,
        wall_time_seconds=1.25,
    )


def _controller() -> FittedController:
    provenance = FitProvenance(
        manifest_sha256=_hash("a"),
        training_draws_sha256=_hash("b"),
        validation_draws_sha256=None,
        implementation_sha256=_hash("c"),
        optimizer_seed=42,
    )
    metadata = FitMetadata(
        method_id="nominal",
        converged=True,
        termination_reason="converged",
        cost=_cost(),
        provenance=provenance,
        objective_contributions=(
            ObjectiveContribution(draw_id="nominal", value=0.5),
        ),
    )
    return FittedController(
        schema_version="1.0",
        n_steps=2,
        max_action=0.25,
        controls=((0.25, 0.0), (0.0, 0.0)),
        metadata=metadata,
    )


def _fit_artifact() -> FitArtifact:
    return FitArtifact(
        schema_version="1.0",
        artifact_type="fit",
        manifest_sha256=_hash("a"),
        training_draws_sha256=_hash("b"),
        implementation_sha256=_hash("c"),
        controller=_controller(),
    )


def _record() -> EvaluationRecord:
    return EvaluationRecord(
        method_id="nominal",
        optimizer_seed=42,
        draw_id="test-b100-0000",
        severity_fraction=0.1,
        final_ergotropy=0.8,
        runtime_seconds=0.05,
        pulse_sha256=_hash("d"),
        fit_artifact_sha256=_hash("e"),
        simulator_sha256=_hash("f"),
    )


def _summary_method() -> MethodSummary:
    return MethodSummary(
        method_id="nominal",
        seed_count=10,
        mean=0.8,
        median=0.81,
        percentile_10=0.7,
        cvar_0_1=0.65,
        minimum=0.6,
        fit_cost=_cost(),
    )


def test_fit_artifact_requires_matching_nested_provenance() -> None:
    assert _fit_artifact().controller.metadata.provenance.optimizer_seed == 42
    with pytest.raises(ValidationError):
        _ = FitArtifact(
            schema_version="1.0",
            artifact_type="fit",
            manifest_sha256=_hash("9"),
            training_draws_sha256=_hash("b"),
            implementation_sha256=_hash("c"),
            controller=_controller(),
        )


def test_raw_results_reject_duplicate_unknown_and_nonfinite_records() -> None:
    record = _record()
    valid = RawResultsArtifact(
        schema_version="1.0",
        artifact_type="raw_results",
        manifest_sha256=_hash("a"),
        evaluation_split="test",
        evaluation_draws_sha256=_hash("1"),
        fit_artifact_sha256s=(_hash("e"),),
        records=(record,),
    )
    assert len(valid.records) == 1
    with pytest.raises(ValidationError):
        _ = RawResultsArtifact(
            schema_version="1.0",
            artifact_type="raw_results",
            manifest_sha256=_hash("a"),
            evaluation_split="test",
            evaluation_draws_sha256=_hash("1"),
            fit_artifact_sha256s=(_hash("e"),),
            records=(record, record),
        )
    with pytest.raises(ValidationError):
        _ = RawResultsArtifact(
            schema_version="1.0",
            artifact_type="raw_results",
            manifest_sha256=_hash("a"),
            evaluation_split="test",
            evaluation_draws_sha256=_hash("1"),
            fit_artifact_sha256s=(_hash("2"),),
            records=(record,),
        )
    payload = record.model_dump()
    payload["final_ergotropy"] = math.nan
    with pytest.raises(ValidationError):
        _ = EvaluationRecord.model_validate(payload)


def test_summary_rejects_duplicate_method_identities() -> None:
    method = _summary_method()
    valid = SummaryArtifact(
        schema_version="1.0",
        artifact_type="summary",
        manifest_sha256=_hash("a"),
        raw_results_sha256=_hash("3"),
        methods=(method,),
    )
    assert valid.methods[0].method_id == "nominal"
    with pytest.raises(ValidationError):
        _ = SummaryArtifact(
            schema_version="1.0",
            artifact_type="summary",
            manifest_sha256=_hash("a"),
            raw_results_sha256=_hash("3"),
            methods=(method, method),
        )


def test_claim_decision_rejects_legacy_and_incoherent_branches() -> None:
    decision = ClaimDecisionArtifact(
        schema_version="1.0",
        artifact_type="claim_decision",
        manifest_sha256=_hash("a"),
        summary_sha256=_hash("3"),
        branch="neutral",
        selected_method_id=None,
        rationale="No superiority gate passed.",
        gates=(GateResult(gate_id="q", passed=False, detail="q >= 0.05"),),
    )
    legacy = decision.model_dump()
    del legacy["schema_version"]
    with pytest.raises(ValidationError):
        _ = ClaimDecisionArtifact.model_validate(legacy)
    with pytest.raises(ValidationError):
        _ = ClaimDecisionArtifact(
            schema_version="1.0",
            artifact_type="claim_decision",
            manifest_sha256=_hash("a"),
            summary_sha256=_hash("3"),
            branch="method-specific",
            selected_method_id=None,
            rationale="Missing method.",
            gates=(),
        )


def test_paper_values_reject_duplicate_keys() -> None:
    value = PaperValue(
        key="nominal_mean",
        value=0.8,
        formatted="0.800",
        source_artifact_sha256=_hash("3"),
        source_selector="methods[nominal].mean",
    )
    valid = PaperValuesArtifact(
        schema_version="1.0",
        artifact_type="paper_values",
        manifest_sha256=_hash("a"),
        summary_sha256=_hash("3"),
        claim_decision_sha256=_hash("4"),
        values=(value,),
    )
    assert valid.values[0].value == 0.8
    with pytest.raises(ValidationError):
        _ = PaperValuesArtifact(
            schema_version="1.0",
            artifact_type="paper_values",
            manifest_sha256=_hash("a"),
            summary_sha256=_hash("3"),
            claim_decision_sha256=_hash("4"),
            values=(value, value),
        )
