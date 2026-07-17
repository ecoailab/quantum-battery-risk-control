"""Resume-safe execution of canonical full controller fits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .artifact_store import ArtifactStore
from .controller import FittedController, TrainingDraws, ValidationDraws
from .full_fit_fitters import FullFitterContext, FullFitterSpec, build_full_fitter
from .full_fit_models import FullFitArtifact, FullFitPlan, FullMethod
from .full_fit_planning import (
    FullFitRequest,
    MissingFullFitConfigError,
    StaleFullFitInputError,
    load_plan_inputs,
    plan_full_fits,
)
from .smoke_artifacts import load_source_draws, read_existing


@dataclass(frozen=True, slots=True)
class FullFitReport:
    fit_hashes: tuple[tuple[str, int, str], ...]
    reused_fit_count: int


@dataclass(frozen=True, slots=True)
class _FitLinks:
    plan: FullFitPlan
    training_sha256: str
    validation_sha256: str
    training_draw_ids: tuple[str, ...]


def run_full_fits(request: FullFitRequest) -> FullFitReport:
    inputs = load_plan_inputs(request)
    plan = plan_full_fits(request)
    training_set = load_source_draws(
        request.canonical_root,
        "train",
        inputs.manifest_sha256,
    )
    validation_set = load_source_draws(
        request.canonical_root,
        "validation",
        inputs.manifest_sha256,
    )
    training_hash = _draw_hash(request.canonical_root, "train")
    validation_hash = _draw_hash(request.canonical_root, "validation")
    training = TrainingDraws(training_set, training_hash)
    context = FullFitterContext(
        request.project_root,
        inputs.manifest,
        inputs.manifest_sha256,
        ValidationDraws(validation_set, validation_hash),
        inputs.config,
    )
    links = _FitLinks(
        plan,
        training_hash,
        validation_hash,
        tuple(draw.draw_id for draw in training_set.draws),
    )
    fits_root = request.canonical_root / "fits"
    store = ArtifactStore(fits_root)
    hashes: list[tuple[str, int, str]] = []
    reused_count = 0
    for identity in plan.identities:
        spec = build_full_fitter(context, identity)
        relative = Path(identity.method_id.value) / f"{identity.optimizer_seed}.json"
        existing, existing_hash = read_existing(
            store,
            fits_root,
            relative,
            FullFitArtifact,
        )
        if existing is not None:
            _validate_existing(
                existing,
                spec,
                links,
            )
            artifact_hash = existing_hash
            reused_count += 1
        else:
            artifact = _fit_artifact(
                spec.fitter.fit(training, identity.optimizer_seed),
                spec,
                links,
            )
            artifact_hash = store.write(relative, artifact).sha256
        hashes.append((identity.method_id.value, identity.optimizer_seed, artifact_hash))
    return FullFitReport(tuple(hashes), reused_count)


def _draw_hash(canonical_root: Path, split: str) -> str:
    return sha256((canonical_root / "draws" / f"{split}.json").read_bytes()).hexdigest()


def _fit_artifact(
    controller: FittedController,
    spec: FullFitterSpec,
    links: _FitLinks,
) -> FullFitArtifact:
    identity = spec.identity
    _validate_contributions(controller, identity.method_id, links.training_draw_ids)
    return FullFitArtifact(
        schema_version="1.0",
        artifact_type="full_fit",
        manifest_sha256=links.plan.manifest_sha256,
        training_draws_sha256=links.training_sha256,
        validation_draws_sha256=(
            links.validation_sha256 if identity.method_id is FullMethod.PPO else None
        ),
        fairness_certificate_sha256=links.plan.fairness_certificate_sha256,
        budget_amendment_sha256=links.plan.budget_amendment_sha256,
        full_fit_config_sha256=links.plan.full_fit_config_sha256,
        implementation_sha256=spec.implementation_sha256,
        budget_kind=identity.budget_kind,
        requested_budget=identity.requested_budget,
        test_data_accessed=False,
        controller=controller,
    )


def _validate_existing(
    artifact: FullFitArtifact,
    spec: FullFitterSpec,
    links: _FitLinks,
) -> None:
    identity = spec.identity
    expected_validation = (
        links.validation_sha256 if identity.method_id is FullMethod.PPO else None
    )
    controller = artifact.controller
    if (
        artifact.manifest_sha256 != links.plan.manifest_sha256
        or artifact.training_draws_sha256 != links.training_sha256
        or artifact.validation_draws_sha256 != expected_validation
        or artifact.fairness_certificate_sha256
        != links.plan.fairness_certificate_sha256
        or artifact.budget_amendment_sha256 != links.plan.budget_amendment_sha256
        or artifact.full_fit_config_sha256 != links.plan.full_fit_config_sha256
        or artifact.implementation_sha256 != spec.implementation_sha256
        or artifact.budget_kind != identity.budget_kind
        or artifact.requested_budget != identity.requested_budget
        or artifact.test_data_accessed
        or controller.metadata.method_id != identity.method_id.value
        or controller.metadata.provenance.optimizer_seed != identity.optimizer_seed
    ):
        raise StaleFullFitInputError(
            f"stale full fit for {identity.method_id.value}/{identity.optimizer_seed}",
        )
    _validate_contributions(controller, identity.method_id, links.training_draw_ids)


def _validate_contributions(
    controller: FittedController,
    method: FullMethod,
    training_draw_ids: tuple[str, ...],
) -> None:
    contribution_ids = tuple(
        item.draw_id for item in controller.metadata.objective_contributions
    )
    expected_by_method = {item: training_draw_ids for item in FullMethod}
    expected_by_method[FullMethod.NOMINAL] = ("nominal",)
    if contribution_ids != expected_by_method[method]:
        raise StaleFullFitInputError(
            "full-fit objective contributions do not match certified split access",
        )


__all__ = [
    "FullFitReport",
    "FullFitRequest",
    "MissingFullFitConfigError",
    "StaleFullFitInputError",
    "plan_full_fits",
    "run_full_fits",
]
