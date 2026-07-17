"""Persistence and resume validation for canonical smoke artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .artifact_models import FitArtifact
from .artifact_store import ArtifactReadRequest, ArtifactStore
from .controller import TrainingDraws
from .fairness import FairnessCertificate, MethodRegistry
from .manifest import CanonicalManifest
from .smoke_fitters import SmokeFitterSpec
from .uncertainty import DrawSet, SplitName, UncertaintyDraw


ModelT = TypeVar("ModelT", bound=BaseModel)


class StaleSmokeArtifactError(ValueError):
    """Raised when an existing smoke artifact violates the current contract."""


def load_fairness(root: Path) -> tuple[MethodRegistry, FairnessCertificate]:
    store = ArtifactStore(root)
    registry, registry_hash = read_existing(
        store,
        root,
        Path("method_registry.json"),
        MethodRegistry,
    )
    certificate, _ = read_existing(
        store,
        root,
        Path("fairness_certificate.json"),
        FairnessCertificate,
    )
    if registry is None or certificate is None:
        raise StaleSmokeArtifactError("certified method registry is required")
    if certificate.registry_sha256 != registry_hash:
        raise StaleSmokeArtifactError("fairness certificate has a stale registry link")
    return registry, certificate


def load_source_draws(root: Path, split: SplitName, manifest_hash: str) -> DrawSet:
    path = root / "draws" / f"{split}.json"
    draw_set = DrawSet.model_validate_json(path.read_bytes())
    if draw_set.split != split or draw_set.source_manifest_sha256 != manifest_hash:
        raise StaleSmokeArtifactError(f"{split} draws do not match the manifest")
    return draw_set


def persist_subset(
    store: ArtifactStore,
    root: Path,
    source: DrawSet,
    count: int,
) -> tuple[DrawSet, str]:
    subset = DrawSet(
        schema_version="1.0",
        split=source.split,
        source_manifest_sha256=source.source_manifest_sha256,
        draws=_representative_draws(source.draws, count),
    )
    relative = Path("draws") / f"{source.split}.json"
    existing, existing_hash = read_existing(store, root, relative, DrawSet)
    if existing is not None:
        if existing != subset:
            raise StaleSmokeArtifactError(f"stale smoke {source.split} draws")
        return existing, existing_hash
    receipt = store.write(relative, subset)
    return subset, receipt.sha256


def fit_or_resume(
    store: ArtifactStore,
    root: Path,
    spec: SmokeFitterSpec,
    training: TrainingDraws,
    validation_hash: str,
    manifest: CanonicalManifest,
    manifest_hash: str,
    seed: int,
) -> tuple[FitArtifact, str, bool]:
    relative = Path("fits") / f"{spec.method_id}.json"
    existing, existing_hash = read_existing(store, root, relative, FitArtifact)
    if existing is not None:
        _validate_fit(existing, spec, training, validation_hash, manifest, seed)
        return existing, existing_hash, True
    controller = spec.fitter.fit(training, seed)
    artifact = FitArtifact(
        schema_version="1.0",
        artifact_type="fit",
        manifest_sha256=manifest_hash,
        training_draws_sha256=training.draw_set_sha256,
        implementation_sha256=spec.implementation_sha256,
        controller=controller,
    )
    receipt = store.write(relative, artifact)
    return artifact, receipt.sha256, False


def read_existing(
    store: ArtifactStore,
    root: Path,
    relative: Path,
    model_type: type[ModelT],
) -> tuple[ModelT | None, str]:
    source = root / relative
    if not source.is_file():
        return None, ""
    artifact_hash = sha256(source.read_bytes()).hexdigest()
    artifact = store.read(model_type, ArtifactReadRequest(relative, artifact_hash))
    return artifact, artifact_hash


def _representative_draws(
    draws: tuple[UncertaintyDraw, ...],
    count: int,
) -> tuple[UncertaintyDraw, ...]:
    selected: list[UncertaintyDraw] = []
    for severity in sorted({draw.severity_fraction for draw in draws}):
        selected.append(next(draw for draw in draws if draw.severity_fraction == severity))
        if len(selected) == count:
            return tuple(selected)
    selected_ids = {draw.draw_id for draw in selected}
    selected.extend(draw for draw in draws if draw.draw_id not in selected_ids)
    return tuple(selected[:count])


def _validate_fit(
    artifact: FitArtifact,
    spec: SmokeFitterSpec,
    training: TrainingDraws,
    validation_hash: str,
    manifest: CanonicalManifest,
    seed: int,
) -> None:
    controller = artifact.controller
    provenance = controller.metadata.provenance
    expected_validation = validation_hash if spec.method_id.startswith("ppo-") else None
    if (
        artifact.manifest_sha256 != training.draw_set.source_manifest_sha256
        or artifact.training_draws_sha256 != training.draw_set_sha256
        or artifact.implementation_sha256 != spec.implementation_sha256
        or controller.metadata.method_id != spec.method_id
        or controller.n_steps != manifest.physics.n_steps
        or controller.max_action != manifest.physics.max_omega
        or provenance.optimizer_seed != seed
        or provenance.validation_draws_sha256 != expected_validation
    ):
        raise StaleSmokeArtifactError(f"stale smoke fit for {spec.method_id}")
