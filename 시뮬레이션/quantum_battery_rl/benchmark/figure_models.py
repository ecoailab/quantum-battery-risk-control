"""Strict sidecars and receipts for canonical publication figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FigureId = Literal[
    "uncertainty_coverage",
    "convergence_cost",
    "heldout_distributions",
    "mean_cvar_comparison",
]


@dataclass(frozen=True, slots=True)
class InvalidFigureSeriesLengthError(ValueError):
    field: str
    expected: int
    actual: int

    @override
    def __str__(self) -> str:
        return (
            f"figure series {self.field} length is {self.actual}; "
            f"expected {self.expected}"
        )


class FigureSeries(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    series_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    panel: Annotated[str, Field(min_length=1)]
    x: tuple[float, ...]
    y: Annotated[tuple[float, ...], Field(min_length=1)]
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()

    @model_validator(mode="after")
    def verify_lengths(self) -> Self:
        if self.x and len(self.x) != len(self.y):
            raise InvalidFigureSeriesLengthError("x", len(self.y), len(self.x))
        if self.lower and len(self.lower) != len(self.y):
            raise InvalidFigureSeriesLengthError("lower", len(self.y), len(self.lower))
        if self.upper and len(self.upper) != len(self.y):
            raise InvalidFigureSeriesLengthError("upper", len(self.y), len(self.upper))
        return self


class FigureSidecar(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["1.0"]
    artifact_type: Literal["figure_sidecar"]
    figure_id: FigureId
    source_artifact_sha256s: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    x_label: Annotated[str, Field(min_length=1)]
    y_label: Annotated[str, Field(min_length=1)]
    series: Annotated[tuple[FigureSeries, ...], Field(min_length=1)]


class FigureReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["1.0"]
    artifact_type: Literal["figure_receipt"]
    figure_implementation_sha256: Sha256
    source_artifact_sha256s: Annotated[tuple[Sha256, ...], Field(min_length=4)]
    output_sha256: dict[str, Sha256]
