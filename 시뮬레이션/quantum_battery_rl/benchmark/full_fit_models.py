"""Strict configuration and artifacts for canonical full controller fits."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .controller import FittedController


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
FullBudgetKind = Literal[
    "objective_evaluations",
    "environment_steps",
    "candidate_evaluations",
]


class DuplicateFullFitIdentityError(ValueError):
    """Raised when a full-fit plan repeats a method/seed identity."""


class FullFitLinkError(ValueError):
    """Raised when nested full-fit provenance disagrees with artifact links."""


class FullMethod(StrEnum):
    NOMINAL = "nominal-lbfgsb-finite-difference"
    MEAN = "saa-mean-lbfgsb-finite-difference"
    CVAR = "cvar-0.1-lbfgsb-finite-difference"
    PPO = "ppo-time-only-domain-randomized"
    BANG_BANG = "bang-bang-ensemble-grid"
    SINUSOIDAL = "sinusoidal-ensemble-grid"
    RANDOM = "random-seeded-reference"


class _FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class FullFitConfigArtifact(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["full_fit_config"]
    manifest_sha256: Sha256
    fairness_certificate_sha256: Sha256
    budget_amendment_sha256: Sha256
    bang_bang_axis_count: PositiveInt
    sinusoidal_frequency_count: PositiveInt
    sinusoidal_phase_count: PositiveInt
    test_data_accessed: Literal[False]


class FullFitIdentity(_FrozenModel):
    method_id: FullMethod
    optimizer_seed: int
    budget_kind: FullBudgetKind
    requested_budget: PositiveInt


class FullFitPlan(_FrozenModel):
    manifest_sha256: Sha256
    fairness_certificate_sha256: Sha256
    budget_amendment_sha256: Sha256
    full_fit_config_sha256: Sha256
    identities: Annotated[tuple[FullFitIdentity, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_unique_identities(self) -> Self:
        keys = {(item.method_id, item.optimizer_seed) for item in self.identities}
        if len(keys) != len(self.identities):
            raise DuplicateFullFitIdentityError
        return self


class FullFitArtifact(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["full_fit"]
    manifest_sha256: Sha256
    training_draws_sha256: Sha256
    validation_draws_sha256: Sha256 | None
    fairness_certificate_sha256: Sha256
    budget_amendment_sha256: Sha256
    full_fit_config_sha256: Sha256
    implementation_sha256: Sha256
    budget_kind: FullBudgetKind
    requested_budget: PositiveInt
    test_data_accessed: Literal[False]
    controller: FittedController

    @model_validator(mode="after")
    def verify_links(self) -> Self:
        provenance = self.controller.metadata.provenance
        expected_validation = (
            self.validation_draws_sha256
            if self.controller.metadata.method_id == FullMethod.PPO
            else None
        )
        if (
            provenance.manifest_sha256 != self.manifest_sha256
            or provenance.training_draws_sha256 != self.training_draws_sha256
            or provenance.validation_draws_sha256 != expected_validation
            or provenance.implementation_sha256 != self.implementation_sha256
        ):
            raise FullFitLinkError
        return self
