"""Input validation and identity planning for canonical full fits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .budget_models import BudgetAmendmentArtifact
from .full_fit_models import (
    FullBudgetKind,
    FullFitConfigArtifact,
    FullFitIdentity,
    FullFitPlan,
    FullMethod,
)
from .manifest import CanonicalManifest, load_manifest
from .smoke_artifacts import load_fairness


class MissingFullFitConfigError(FileNotFoundError):
    """Raised when no pre-test full-fit configuration has been declared."""

    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Full-fit configuration does not exist: {path}")


class StaleFullFitInputError(ValueError):
    """Raised when a full-fit input does not match the current evidence graph."""


class UnknownFullFitMethodError(ValueError):
    """Raised when an operational shard names an uncertified method."""

    method_id: str

    def __init__(self, method_id: str) -> None:
        self.method_id = method_id
        super().__init__(f"Unknown full-fit method: {method_id}")


class UndeclaredOptimizerSeedError(ValueError):
    """Raised when an operational shard names an undeclared optimizer seed."""

    seed: int

    def __init__(self, seed: int) -> None:
        self.seed = seed
        super().__init__(f"Undeclared optimizer seed: {seed}")


@dataclass(frozen=True, slots=True)
class FullFitRequest:
    project_root: Path
    manifest_path: Path
    canonical_root: Path
    method_ids: tuple[str, ...] | None = None
    optimizer_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class PlanInputs:
    manifest: CanonicalManifest
    manifest_sha256: str
    config: FullFitConfigArtifact
    config_sha256: str
    fairness_sha256: str
    amendment: BudgetAmendmentArtifact
    amendment_sha256: str
    registered_method_ids: tuple[str, ...]


def plan_full_fits(request: FullFitRequest) -> FullFitPlan:
    inputs = load_plan_inputs(request)
    methods = _select_methods(request, inputs.registered_method_ids)
    seeds = _select_seeds(request, inputs.manifest.optimization.optimizer_seeds)
    selected_budgets = _selected_budgets(inputs.amendment)
    identities = tuple(
        _identity(method, seed, inputs, selected_budgets)
        for method in methods
        for seed in seeds
    )
    return FullFitPlan(
        manifest_sha256=inputs.manifest_sha256,
        fairness_certificate_sha256=inputs.fairness_sha256,
        budget_amendment_sha256=inputs.amendment_sha256,
        full_fit_config_sha256=inputs.config_sha256,
        identities=identities,
    )


def load_plan_inputs(request: FullFitRequest) -> PlanInputs:
    manifest_payload = request.manifest_path.read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = load_manifest(request.manifest_path)
    registry, certificate = load_fairness(request.canonical_root)
    registered = tuple(method.method_id for method in registry.methods)
    expected_methods = tuple(method.value for method in FullMethod)
    if (
        not certificate.certified
        or certificate.manifest_sha256 != manifest_hash
        or registered != expected_methods
    ):
        raise StaleFullFitInputError("full fits require the current certified registry")
    fairness_path = request.canonical_root / "fairness_certificate.json"
    fairness_hash = sha256(fairness_path.read_bytes()).hexdigest()
    amendment_path = request.canonical_root / "pilot" / "budget_amendment.json"
    amendment_payload = amendment_path.read_bytes()
    amendment_hash = sha256(amendment_payload).hexdigest()
    amendment = BudgetAmendmentArtifact.model_validate_json(amendment_payload)
    config_path = request.canonical_root / "full_fit_config.json"
    if not config_path.is_file():
        raise MissingFullFitConfigError(config_path)
    config_payload = config_path.read_bytes()
    config_hash = sha256(config_payload).hexdigest()
    config = FullFitConfigArtifact.model_validate_json(config_payload)
    if (
        amendment.manifest_sha256 != manifest_hash
        or amendment.fairness_certificate_sha256 != fairness_hash
        or amendment.test_data_accessed
        or config.manifest_sha256 != manifest_hash
        or config.fairness_certificate_sha256 != fairness_hash
        or config.budget_amendment_sha256 != amendment_hash
        or config.test_data_accessed
    ):
        raise StaleFullFitInputError("full-fit inputs have stale provenance")
    return PlanInputs(
        manifest,
        manifest_hash,
        config,
        config_hash,
        fairness_hash,
        amendment,
        amendment_hash,
        registered,
    )


def _select_methods(
    request: FullFitRequest,
    registered: tuple[str, ...],
) -> tuple[FullMethod, ...]:
    requested = request.method_ids if request.method_ids is not None else registered
    methods: list[FullMethod] = []
    for method_id in requested:
        try:
            method = FullMethod(method_id)
        except ValueError as error:
            raise UnknownFullFitMethodError(method_id) from error
        if method.value not in registered:
            raise UnknownFullFitMethodError(method_id)
        methods.append(method)
    return tuple(methods)


def _select_seeds(
    request: FullFitRequest,
    declared: tuple[int, ...],
) -> tuple[int, ...]:
    seeds = request.optimizer_seeds if request.optimizer_seeds is not None else declared
    for seed in seeds:
        if seed not in declared:
            raise UndeclaredOptimizerSeedError(seed)
    return seeds


def _selected_budgets(
    amendment: BudgetAmendmentArtifact,
) -> dict[FullMethod, int]:
    budgets: dict[FullMethod, int] = {}
    for decision in amendment.decisions:
        try:
            method = FullMethod(decision.method_id)
        except ValueError as error:
            raise StaleFullFitInputError("budget amendment names an unknown method") from error
        if decision.status != "plateau" or decision.selected_budget is None:
            raise StaleFullFitInputError("every optimized method requires a selected budget")
        budgets[method] = decision.selected_budget
    required = {FullMethod.NOMINAL, FullMethod.MEAN, FullMethod.CVAR, FullMethod.PPO}
    if set(budgets) != required:
        raise StaleFullFitInputError("budget amendment has an incomplete optimized method set")
    return budgets


def _identity(
    method: FullMethod,
    seed: int,
    inputs: PlanInputs,
    selected_budgets: dict[FullMethod, int],
) -> FullFitIdentity:
    budgets = {
        FullMethod.NOMINAL: selected_budgets[FullMethod.NOMINAL],
        FullMethod.MEAN: selected_budgets[FullMethod.MEAN],
        FullMethod.CVAR: selected_budgets[FullMethod.CVAR],
        FullMethod.PPO: selected_budgets[FullMethod.PPO],
        FullMethod.BANG_BANG: (
            inputs.manifest.physics.n_steps * inputs.config.bang_bang_axis_count
        ),
        FullMethod.SINUSOIDAL: (
            inputs.config.sinusoidal_frequency_count
            * inputs.config.sinusoidal_phase_count
        ),
        FullMethod.RANDOM: 1,
    }
    kinds: dict[FullMethod, FullBudgetKind] = {
        FullMethod.NOMINAL: "objective_evaluations",
        FullMethod.MEAN: "objective_evaluations",
        FullMethod.CVAR: "objective_evaluations",
        FullMethod.PPO: "environment_steps",
        FullMethod.BANG_BANG: "candidate_evaluations",
        FullMethod.SINUSOIDAL: "candidate_evaluations",
        FullMethod.RANDOM: "candidate_evaluations",
    }
    return FullFitIdentity(
        method_id=method,
        optimizer_seed=seed,
        budget_kind=kinds[method],
        requested_budget=budgets[method],
    )
