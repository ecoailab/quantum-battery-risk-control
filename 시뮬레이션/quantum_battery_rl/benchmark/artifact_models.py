"""Versioned canonical benchmark artifact models."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .controller import FitCost, FittedController


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
UnitValue = Annotated[float, Field(ge=0.0, le=1.0)]
UnitFraction = Annotated[float, Field(gt=0.0, lt=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class ArtifactLinkError(ValueError):
    """Raised when nested provenance disagrees with an artifact link."""


class DuplicateArtifactEntryError(ValueError):
    """Raised when a canonical collection contains duplicate identities."""


class _ArtifactModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class FitArtifact(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["fit"]
    manifest_sha256: Sha256
    training_draws_sha256: Sha256
    implementation_sha256: Sha256
    controller: FittedController

    @model_validator(mode="after")
    def verify_links(self) -> Self:
        provenance = self.controller.metadata.provenance
        expected = (
            self.manifest_sha256,
            self.training_draws_sha256,
            self.implementation_sha256,
        )
        actual = (
            provenance.manifest_sha256,
            provenance.training_draws_sha256,
            provenance.implementation_sha256,
        )
        if actual != expected:
            raise ArtifactLinkError("Fit provenance does not match artifact links")
        return self


class EvaluationRecord(_ArtifactModel):
    method_id: NonEmptyString
    optimizer_seed: int
    draw_id: NonEmptyString
    severity_fraction: UnitFraction
    final_ergotropy: UnitValue
    runtime_seconds: NonNegativeFloat
    pulse_sha256: Sha256
    fit_artifact_sha256: Sha256
    simulator_sha256: Sha256


class RawResultsArtifact(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["raw_results"]
    manifest_sha256: Sha256
    evaluation_split: Literal["validation", "test"]
    evaluation_draws_sha256: Sha256
    fit_artifact_sha256s: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    records: Annotated[tuple[EvaluationRecord, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_records(self) -> Self:
        linked_fits = set(self.fit_artifact_sha256s)
        if len(linked_fits) != len(self.fit_artifact_sha256s):
            raise DuplicateArtifactEntryError("Duplicate fit artifact hash")
        identities = {
            (record.method_id, record.optimizer_seed, record.draw_id)
            for record in self.records
        }
        if len(identities) != len(self.records):
            raise DuplicateArtifactEntryError("Duplicate raw evaluation identity")
        if any(record.fit_artifact_sha256 not in linked_fits for record in self.records):
            raise ArtifactLinkError("Raw evaluation references an unlinked fit artifact")
        return self


class HeldOutResultsArtifact(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["held_out_raw_results"]
    manifest_sha256: Sha256
    evaluation_split: Literal["test"]
    evaluation_draws_sha256: Sha256
    fairness_certificate_sha256: Sha256
    full_fit_config_sha256: Sha256
    evaluation_implementation_sha256: Sha256
    simulator_sha256: Sha256
    fit_artifact_sha256s: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    records: Annotated[tuple[EvaluationRecord, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_records(self) -> Self:
        linked_fits = set(self.fit_artifact_sha256s)
        identities = {
            (record.method_id, record.optimizer_seed, record.draw_id)
            for record in self.records
        }
        if len(linked_fits) != len(self.fit_artifact_sha256s):
            raise DuplicateArtifactEntryError("Duplicate held-out fit artifact hash")
        if len(identities) != len(self.records):
            raise DuplicateArtifactEntryError("Duplicate held-out evaluation identity")
        if any(record.fit_artifact_sha256 not in linked_fits for record in self.records):
            raise ArtifactLinkError("Held-out evaluation references an unlinked fit")
        if any(record.simulator_sha256 != self.simulator_sha256 for record in self.records):
            raise ArtifactLinkError("Held-out evaluation has a stale simulator link")
        return self


class HeldOutEvaluationReceipt(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["held_out_evaluation_receipt"]
    manifest_sha256: Sha256
    raw_results_sha256: Sha256
    evaluation_implementation_sha256: Sha256


class MethodSummary(_ArtifactModel):
    method_id: NonEmptyString
    seed_count: PositiveInt
    mean: UnitValue
    median: UnitValue
    percentile_10: UnitValue
    cvar_0_1: UnitValue
    minimum: UnitValue
    fit_cost: FitCost


class SummaryArtifact(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["summary"]
    manifest_sha256: Sha256
    raw_results_sha256: Sha256
    methods: Annotated[tuple[MethodSummary, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_methods(self) -> Self:
        method_ids = {method.method_id for method in self.methods}
        if len(method_ids) != len(self.methods):
            raise DuplicateArtifactEntryError("Duplicate summary method identity")
        return self


class GateResult(_ArtifactModel):
    gate_id: NonEmptyString
    passed: bool
    detail: NonEmptyString


class ClaimDecisionArtifact(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["claim_decision"]
    manifest_sha256: Sha256
    summary_sha256: Sha256
    branch: Literal["comparative", "method-specific", "neutral"]
    selected_method_id: NonEmptyString | None
    comparator_method_id: NonEmptyString | None = None
    rationale: NonEmptyString
    gates: tuple[GateResult, ...]

    @model_validator(mode="after")
    def verify_decision(self) -> Self:
        gate_ids = {gate.gate_id for gate in self.gates}
        if len(gate_ids) != len(self.gates):
            raise DuplicateArtifactEntryError("Duplicate claim gate identity")
        if self.branch == "method-specific" and self.selected_method_id is None:
            raise ArtifactLinkError("Method-specific branch requires a selected method")
        if self.branch != "method-specific" and self.selected_method_id is not None:
            raise ArtifactLinkError("Only a method-specific branch may select a method")
        return self


class PaperValue(_ArtifactModel):
    key: NonEmptyString
    value: float
    formatted: NonEmptyString
    source_artifact_sha256: Sha256
    source_selector: NonEmptyString


class PaperValuesArtifact(_ArtifactModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["paper_values"]
    manifest_sha256: Sha256
    summary_sha256: Sha256
    claim_decision_sha256: Sha256
    values: Annotated[tuple[PaperValue, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_values(self) -> Self:
        keys = {value.key for value in self.values}
        if len(keys) != len(self.values):
            raise DuplicateArtifactEntryError("Duplicate paper value key")
        return self
