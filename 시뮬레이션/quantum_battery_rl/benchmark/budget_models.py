"""Strict convergence-pilot and budget-amendment artifacts."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .controller import FittedController


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
UnitValue = Annotated[float, Field(ge=0.0, le=1.0)]
BudgetKind = Literal["objective_evaluations", "environment_steps"]
DecisionStatus = Literal["plateau", "extension-required"]


class _FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class PlateauCriterion(_FrozenModel):
    absolute_tolerance: Annotated[float, Field(gt=0.0)]
    required_consecutive_transitions: Annotated[int, Field(ge=2)]
    extension_multiplier: Annotated[int, Field(ge=2)]
    maximum_extensions: NonNegativeInt


class PilotFitArtifact(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["pilot_fit"]
    manifest_sha256: Sha256
    training_draws_sha256: Sha256
    validation_draws_sha256: Sha256
    implementation_sha256: Sha256
    budget_kind: BudgetKind
    requested_budget: PositiveInt
    controller: FittedController

    @model_validator(mode="after")
    def verify_links(self) -> Self:
        provenance = self.controller.metadata.provenance
        if (
            provenance.manifest_sha256 != self.manifest_sha256
            or provenance.training_draws_sha256 != self.training_draws_sha256
            or provenance.implementation_sha256 != self.implementation_sha256
        ):
            raise ValueError("pilot fit provenance does not match artifact links")
        expected_validation = (
            self.validation_draws_sha256
            if self.controller.metadata.method_id.startswith("ppo-")
            else None
        )
        if provenance.validation_draws_sha256 != expected_validation:
            raise ValueError("pilot fit has invalid validation access")
        return self


class PilotStageResult(_FrozenModel):
    method_id: Annotated[str, Field(min_length=1)]
    budget_kind: BudgetKind
    requested_budget: PositiveInt
    objective_evaluations: NonNegativeInt
    environment_steps: NonNegativeInt
    validation_mean: UnitValue
    fit_artifact_sha256: Sha256


class BudgetDecision(_FrozenModel):
    method_id: Annotated[str, Field(min_length=1)]
    budget_kind: BudgetKind
    status: DecisionStatus
    observed_budgets: Annotated[tuple[PositiveInt, ...], Field(min_length=3)]
    last_absolute_changes: Annotated[tuple[float, ...], Field(min_length=2)]
    selected_budget: PositiveInt | None
    next_budget: PositiveInt | None

    @model_validator(mode="after")
    def verify_status(self) -> Self:
        if tuple(sorted(set(self.observed_budgets))) != self.observed_budgets:
            raise ValueError("observed budgets must be unique and increasing")
        if self.status == "plateau":
            if self.selected_budget != self.observed_budgets[-1] or self.next_budget is not None:
                raise ValueError("plateau must select the latest observed budget")
        elif self.selected_budget is not None or self.next_budget is None:
            raise ValueError("extension-required must declare only the next budget")
        return self


class BudgetAmendmentArtifact(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["budget_amendment"]
    manifest_sha256: Sha256
    training_draws_sha256: Sha256
    validation_draws_sha256: Sha256
    fairness_certificate_sha256: Sha256
    pilot_implementation_sha256: Sha256
    criterion: PlateauCriterion
    stages: Annotated[tuple[PilotStageResult, ...], Field(min_length=1)]
    decisions: Annotated[tuple[BudgetDecision, ...], Field(min_length=1)]
    test_data_accessed: Literal[False]

    @model_validator(mode="after")
    def verify_graph(self) -> Self:
        stage_keys = {
            (stage.method_id, stage.budget_kind, stage.requested_budget)
            for stage in self.stages
        }
        if len(stage_keys) != len(self.stages):
            raise ValueError("duplicate pilot stage")
        decision_ids = {decision.method_id for decision in self.decisions}
        if len(decision_ids) != len(self.decisions):
            raise ValueError("duplicate budget decision")
        if decision_ids != {stage.method_id for stage in self.stages}:
            raise ValueError("every staged method requires one budget decision")
        for decision in self.decisions:
            budgets = tuple(
                stage.requested_budget
                for stage in self.stages
                if stage.method_id == decision.method_id
            )
            if budgets != decision.observed_budgets:
                raise ValueError("budget decision does not match staged evidence")
        return self


def assess_budget(
    stages: tuple[PilotStageResult, ...],
    criterion: PlateauCriterion,
) -> BudgetDecision:
    required = criterion.required_consecutive_transitions
    if len(stages) < required + 1:
        raise ValueError("insufficient stages for the plateau criterion")
    method_ids = {stage.method_id for stage in stages}
    budget_kinds = {stage.budget_kind for stage in stages}
    budgets = tuple(stage.requested_budget for stage in stages)
    if len(method_ids) != 1 or len(budget_kinds) != 1:
        raise ValueError("one method and budget kind are required")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("stage budgets must be unique and increasing")
    changes = tuple(
        abs(current.validation_mean - previous.validation_mean)
        for previous, current in zip(stages[-required - 1 : -1], stages[-required:], strict=True)
    )
    plateau = all(change <= criterion.absolute_tolerance for change in changes)
    latest = budgets[-1]
    return BudgetDecision(
        method_id=stages[0].method_id,
        budget_kind=stages[0].budget_kind,
        status="plateau" if plateau else "extension-required",
        observed_budgets=budgets,
        last_absolute_changes=changes,
        selected_budget=latest if plateau else None,
        next_budget=None if plateau else latest * criterion.extension_multiplier,
    )
