"""Canonical controller registry and fairness audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .manifest import CanonicalManifest


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]
FitAccess = Literal["nominal", "train", "validation"]

EXPECTED_ACCESS: Final[dict[str, tuple[FitAccess, ...]]] = {
    "nominal-lbfgsb-finite-difference": ("nominal",),
    "saa-mean-lbfgsb-finite-difference": ("train",),
    "cvar-0.1-lbfgsb-finite-difference": ("train",),
    "ppo-time-only-domain-randomized": ("train", "validation"),
    "bang-bang-ensemble-grid": ("train",),
    "sinusoidal-ensemble-grid": ("train",),
    "random-seeded-reference": ("train",),
}
CANONICAL_SIMULATOR_SOURCES: Final = (
    Path("quantum_battery_rl/env/quantum_dynamics.py"),
    Path("quantum_battery_rl/env/lindblad_env.py"),
    Path("quantum_battery_rl/agents/baselines.py"),
)


class DuplicateMethodRegistrationError(ValueError):
    """Raised when a method identifier appears more than once."""


class _FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class MethodRegistration(_FrozenModel):
    method_id: NonEmptyString
    deployment_observation: Literal["normalized_time"]
    fitting_split_access: tuple[FitAccess, ...]
    test_split_access: bool
    n_steps: Annotated[int, Field(gt=0)]
    max_action: Annotated[float, Field(gt=0.0)]
    metric: Literal["normalized_ergotropy"]
    initial_state: Literal["ground"]
    simulator_sha256: Sha256
    optimizer_seeds: Annotated[tuple[int, ...], Field(min_length=1)]
    frozen_before_test: bool
    cost_accounting: Literal["reported"]


class MethodRegistry(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["method_registry"]
    manifest_sha256: Sha256
    methods: Annotated[tuple[MethodRegistration, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def verify_unique_methods(self) -> Self:
        method_ids = tuple(method.method_id for method in self.methods)
        if len(method_ids) != len(set(method_ids)):
            raise DuplicateMethodRegistrationError("method identifiers must be unique")
        return self


class FairnessCheck(_FrozenModel):
    check_id: NonEmptyString
    passed: bool
    detail: NonEmptyString


class FairnessCertificate(_FrozenModel):
    schema_version: Literal["1.0"]
    artifact_type: Literal["fairness_certificate"]
    manifest_sha256: Sha256
    registry_sha256: Sha256
    method_ids: tuple[NonEmptyString, ...]
    checks: Annotated[tuple[FairnessCheck, ...], Field(min_length=1)]
    certified: bool


def build_default_registry(
    manifest: CanonicalManifest,
    manifest_sha256: str,
    simulator_sha256: str,
) -> MethodRegistry:
    methods = tuple(
        MethodRegistration(
            method_id=method_id,
            deployment_observation="normalized_time",
            fitting_split_access=access,
            test_split_access=False,
            n_steps=manifest.physics.n_steps,
            max_action=manifest.physics.max_omega,
            metric="normalized_ergotropy",
            initial_state="ground",
            simulator_sha256=simulator_sha256,
            optimizer_seeds=manifest.optimization.optimizer_seeds,
            frozen_before_test=True,
            cost_accounting="reported",
        )
        for method_id, access in EXPECTED_ACCESS.items()
    )
    return MethodRegistry(
        schema_version="1.0",
        artifact_type="method_registry",
        manifest_sha256=manifest_sha256,
        methods=methods,
    )


def audit_registry(registry: MethodRegistry, manifest: CanonicalManifest) -> FairnessCertificate:
    method_ids = tuple(method.method_id for method in registry.methods)
    expected_ids = tuple(EXPECTED_ACCESS)
    simulator_hashes = {method.simulator_sha256 for method in registry.methods}
    checks = (
        _check(
            "complete_method_set",
            len(method_ids) == len(expected_ids) and set(method_ids) == set(expected_ids),
            "registry contains exactly the seven declared benchmark methods",
        ),
        _check(
            "allowed_fitting_access",
            all(
                EXPECTED_ACCESS.get(method.method_id) == method.fitting_split_access
                for method in registry.methods
            ),
            "each method uses only its declared fitting splits",
        ),
        _check(
            "no_test_access",
            all(not method.test_split_access for method in registry.methods),
            "test draws remain sealed until all pulses are frozen",
        ),
        _check(
            "shared_horizon",
            all(method.n_steps == manifest.physics.n_steps for method in registry.methods),
            "all methods use the canonical control horizon",
        ),
        _check(
            "shared_action_bounds",
            all(method.max_action == manifest.physics.max_omega for method in registry.methods),
            "all methods use the canonical action bound",
        ),
        _check(
            "shared_metric",
            all(method.metric == "normalized_ergotropy" for method in registry.methods),
            "all methods optimize and report normalized ergotropy",
        ),
        _check(
            "shared_initial_state",
            all(method.initial_state == "ground" for method in registry.methods),
            "all methods start from the ground state",
        ),
        _check(
            "shared_simulator",
            len(simulator_hashes) == 1,
            "all methods use one content-addressed simulator implementation",
        ),
        _check(
            "shared_optimizer_seeds",
            all(
                method.optimizer_seeds == manifest.optimization.optimizer_seeds
                for method in registry.methods
            ),
            "all methods receive the canonical optimizer seed list",
        ),
        _check(
            "time_only_information",
            all(
                method.deployment_observation == "normalized_time"
                for method in registry.methods
            ),
            "deployed controllers receive normalized time only",
        ),
        _check(
            "frozen_before_test",
            all(method.frozen_before_test for method in registry.methods),
            "every pulse is frozen before test evaluation",
        ),
        _check(
            "reported_compute_cost",
            all(method.cost_accounting == "reported" for method in registry.methods),
            "actual fitting cost is reported rather than artificially equalized",
        ),
    )
    registry_payload = (registry.model_dump_json(indent=2) + "\n").encode("utf-8")
    registry_sha256 = hashlib.sha256(registry_payload).hexdigest()
    return FairnessCertificate(
        schema_version="1.0",
        artifact_type="fairness_certificate",
        manifest_sha256=registry.manifest_sha256,
        registry_sha256=registry_sha256,
        method_ids=method_ids,
        checks=checks,
        certified=all(check.passed for check in checks),
    )


def _check(check_id: str, passed: bool, detail: str) -> FairnessCheck:
    return FairnessCheck(check_id=check_id, passed=passed, detail=detail)


def canonical_simulator_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in CANONICAL_SIMULATOR_SOURCES:
        source = project_root / relative_path
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
