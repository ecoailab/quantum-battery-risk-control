"""Synthetic quasi-static uncertainty draw contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

from .manifest import CanonicalManifest


SplitName = Literal["train", "validation", "test"]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_DIFFERENTIAL_SHARE = 0.4
_MAX_ATTEMPTS_PER_DRAW = 100


class SamplingExhaustedError(RuntimeError):
    """Raised when a physical ensemble cannot be sampled from its bounds."""


class UncertaintyDraw(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    draw_id: str
    split: SplitName
    severity_fraction: float
    t1: float
    t2: float
    common_mode: float
    differential_mode: float


class DrawSet(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["1.0"]
    split: SplitName
    source_manifest_sha256: Sha256
    draws: tuple[UncertaintyDraw, ...]


@dataclass(frozen=True, slots=True)
class _GenerationRequest:
    manifest: CanonicalManifest
    split: SplitName
    seed: int
    scenarios_per_band: int
    source_manifest_sha256: str


def generate_draw_sets(
    manifest: CanonicalManifest,
    source_manifest_sha256: str,
) -> tuple[DrawSet, DrawSet, DrawSet]:
    requests = (
        _GenerationRequest(
            manifest,
            "train",
            manifest.splits.seeds.train,
            manifest.splits.scenarios.train,
            source_manifest_sha256,
        ),
        _GenerationRequest(
            manifest,
            "validation",
            manifest.splits.seeds.validation,
            manifest.splits.scenarios.validation,
            source_manifest_sha256,
        ),
        _GenerationRequest(
            manifest,
            "test",
            manifest.splits.seeds.test,
            manifest.splits.scenarios.test,
            source_manifest_sha256,
        ),
    )
    return (
        _generate_split(requests[0]),
        _generate_split(requests[1]),
        _generate_split(requests[2]),
    )


def write_draw_set(draw_set: DrawSet, output_root: Path) -> Path:
    draw_directory = output_root / "draws"
    draw_directory.mkdir(parents=True, exist_ok=True)
    destination = draw_directory / f"{draw_set.split}.json"
    temporary = destination.with_suffix(".json.tmp")
    payload = draw_set.model_dump_json(indent=2) + "\n"
    _ = temporary.write_text(payload, encoding="utf-8", newline="\n")
    _ = temporary.replace(destination)
    return destination


def _generate_split(request: _GenerationRequest) -> DrawSet:
    rng = np.random.default_rng(request.seed)
    draws = tuple(
        draw
        for severity in request.manifest.uncertainty.severity_fractions
        for draw in _sample_band(request, rng, severity)
    )
    return DrawSet(
        schema_version="1.0",
        split=request.split,
        source_manifest_sha256=request.source_manifest_sha256,
        draws=draws,
    )


def _sample_band(
    request: _GenerationRequest,
    rng: Generator,
    severity: float,
) -> tuple[UncertaintyDraw, ...]:
    accepted: list[UncertaintyDraw] = []
    max_attempts = request.scenarios_per_band * _MAX_ATTEMPTS_PER_DRAW
    attempts = 0
    while len(accepted) < request.scenarios_per_band and attempts < max_attempts:
        attempts += 1
        common_span = severity * (1.0 - _DIFFERENTIAL_SHARE)
        differential_span = severity * _DIFFERENTIAL_SHARE
        common = float(rng.uniform(-common_span, common_span))
        differential = float(rng.uniform(-differential_span, differential_span))
        t1 = request.manifest.physics.t1 * (1.0 + common + differential)
        t2 = request.manifest.physics.t2 * (1.0 + common - differential)
        if t2 > 2.0 * t1:
            continue
        accepted.append(
            UncertaintyDraw(
                draw_id=_draw_id(request.split, severity, len(accepted)),
                split=request.split,
                severity_fraction=severity,
                t1=t1,
                t2=t2,
                common_mode=common,
                differential_mode=differential,
            ),
        )
    if len(accepted) != request.scenarios_per_band:
        message = (
            f"Could not sample {request.scenarios_per_band} physical draws for "
            f"{request.split} at severity {severity}."
        )
        raise SamplingExhaustedError(message)
    return tuple(accepted)


def _draw_id(split: SplitName, severity: float, index: int) -> str:
    band = round(severity * 1_000)
    return f"{split}-b{band:03d}-{index:04d}"
