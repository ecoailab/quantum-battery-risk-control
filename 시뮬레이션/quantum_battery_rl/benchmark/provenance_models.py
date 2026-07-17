"""Strict publication-provenance artifact models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from .figure_models import FigureId


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]


@dataclass(frozen=True, slots=True)
class InvalidProvenanceArtifactError(ValueError):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


class _ProvenanceModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class PublicationValueMapping(_ProvenanceModel):
    key: NonEmptyString
    value: float
    formatted: NonEmptyString
    source_artifact_sha256: Sha256
    source_selector: NonEmptyString


class GeneratedFragmentMapping(_ProvenanceModel):
    path: NonEmptyString
    sha256: Sha256
    source_value_keys: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]


class PlotSeriesMapping(_ProvenanceModel):
    figure_id: FigureId
    sidecar_path: NonEmptyString
    sidecar_sha256: Sha256
    series_id: NonEmptyString
    panel: NonEmptyString
    source_artifact_sha256s: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    point_count: Annotated[int, Field(ge=1)]
    data_sha256: Sha256


class PublicationProvenanceArtifact(_ProvenanceModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["publication_provenance"]
    paper_values_receipt_sha256: Sha256
    paper_values_sha256: Sha256
    figure_receipt_sha256: Sha256
    publication_source_sha256: dict[NonEmptyString, Sha256]
    value_mapping_count: Annotated[int, Field(ge=1)]
    fragment_count: Annotated[int, Field(ge=1)]
    figure_count: Annotated[int, Field(ge=1)]
    plot_point_count: Annotated[int, Field(ge=1)]
    value_mappings: Annotated[
        tuple[PublicationValueMapping, ...],
        Field(min_length=1),
    ]
    fragments: Annotated[tuple[GeneratedFragmentMapping, ...], Field(min_length=1)]
    plot_series: Annotated[tuple[PlotSeriesMapping, ...], Field(min_length=1)]
    unmapped_inputs: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    legacy_inputs: tuple[str, ...]

    @model_validator(mode="after")
    def verify_counts_and_uniqueness(self) -> Self:
        if self.value_mapping_count != len(self.value_mappings):
            raise InvalidProvenanceArtifactError(
                "value mapping count does not match mappings"
            )
        if self.fragment_count != len(self.fragments):
            raise InvalidProvenanceArtifactError(
                "fragment count does not match fragments"
            )
        if self.figure_count != len({item.figure_id for item in self.plot_series}):
            raise InvalidProvenanceArtifactError(
                "figure count does not match plot series"
            )
        if self.plot_point_count != sum(item.point_count for item in self.plot_series):
            raise InvalidProvenanceArtifactError(
                "plot point count does not match plot series"
            )
        if len({item.key for item in self.value_mappings}) != len(self.value_mappings):
            raise InvalidProvenanceArtifactError(
                "publication value mappings contain duplicate keys"
            )
        if len({item.path for item in self.fragments}) != len(self.fragments):
            raise InvalidProvenanceArtifactError(
                "generated fragment mappings contain duplicate paths"
            )
        return self
