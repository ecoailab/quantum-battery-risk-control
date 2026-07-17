from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from quantum_battery_rl.benchmark import (
    FairnessCertificate,
    MethodRegistry,
    audit_registry,
    build_default_registry,
)
from quantum_battery_rl.benchmark.artifact_store import ArtifactReadRequest, ArtifactStore
from quantum_battery_rl.benchmark.manifest import load_manifest


MANIFEST_PATH: Final = Path(__file__).parents[1] / "canonical_manifest.json"
MANIFEST_HASH: Final = "a0673ec163fbc5bf1617e59ee8cc0e87d81e60f9994e91542206106b89259464"
SIMULATOR_HASH: Final = "f" * 64
EXPECTED_METHODS: Final = {
    "nominal-lbfgsb-finite-difference",
    "saa-mean-lbfgsb-finite-difference",
    "cvar-0.1-lbfgsb-finite-difference",
    "ppo-time-only-domain-randomized",
    "bang-bang-ensemble-grid",
    "sinusoidal-ensemble-grid",
    "random-seeded-reference",
}


def _registry() -> MethodRegistry:
    return build_default_registry(
        load_manifest(MANIFEST_PATH),
        MANIFEST_HASH,
        SIMULATOR_HASH,
    )


def test_default_registry_certifies_all_seven_methods() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    registry = _registry()

    certificate = audit_registry(registry, manifest)

    assert {method.method_id for method in registry.methods} == EXPECTED_METHODS
    assert certificate.certified
    assert all(check.passed for check in certificate.checks)
    assert all(method.deployment_observation == "normalized_time" for method in registry.methods)
    assert all(not method.test_split_access for method in registry.methods)
    assert all(method.cost_accounting == "reported" for method in registry.methods)
    ppo = next(method for method in registry.methods if method.method_id.startswith("ppo-"))
    assert ppo.fitting_split_access == ("train", "validation")


def test_audit_blocks_privileged_or_mismatched_methods() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    registry = _registry()
    privileged = registry.methods[0].model_copy(update={"test_split_access": True})
    wrong_horizon = registry.methods[1].model_copy(update={"n_steps": 1})
    mutated = MethodRegistry(
        schema_version="1.0",
        artifact_type="method_registry",
        manifest_sha256=registry.manifest_sha256,
        methods=(privileged, wrong_horizon, *registry.methods[2:]),
    )

    certificate = audit_registry(mutated, manifest)

    assert not certificate.certified
    failed = {check.check_id for check in certificate.checks if not check.passed}
    assert {"no_test_access", "shared_horizon"}.issubset(failed)


@pytest.mark.parametrize(
    ("update", "failed_check"),
    [
        ({"fitting_split_access": ("train", "validation")}, "allowed_fitting_access"),
        ({"max_action": 0.5}, "shared_action_bounds"),
        ({"metric": "legacy_energy"}, "shared_metric"),
        ({"initial_state": "excited"}, "shared_initial_state"),
        ({"simulator_sha256": "e" * 64}, "shared_simulator"),
        ({"optimizer_seeds": (999,)}, "shared_optimizer_seeds"),
        ({"deployment_observation": "full_state"}, "time_only_information"),
        ({"frozen_before_test": False}, "frozen_before_test"),
        ({"cost_accounting": "equalized"}, "reported_compute_cost"),
    ],
)
def test_audit_rejects_each_unfair_contract(
    update: dict[str, object],
    failed_check: str,
) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    registry = _registry()
    mutated_method = registry.methods[0].model_copy(update=update)
    mutated_registry = registry.model_copy(
        update={"methods": (mutated_method, *registry.methods[1:])},
    )

    certificate = audit_registry(mutated_registry, manifest)

    assert not certificate.certified
    assert failed_check in {
        check.check_id for check in certificate.checks if not check.passed
    }


def test_registry_rejects_duplicate_and_audit_rejects_missing_method() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    registry = _registry()
    with pytest.raises(ValidationError, match="method identifiers must be unique"):
        _ = MethodRegistry(
            schema_version="1.0",
            artifact_type="method_registry",
            manifest_sha256=registry.manifest_sha256,
            methods=(*registry.methods, registry.methods[0]),
        )

    incomplete = registry.model_copy(update={"methods": registry.methods[:-1]})
    certificate = audit_registry(incomplete, manifest)

    assert not certificate.certified
    assert not next(
        check.passed
        for check in certificate.checks
        if check.check_id == "complete_method_set"
    )


def test_certificate_persistence_is_deterministic(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    registry = _registry()
    certificate = audit_registry(registry, manifest)
    store = ArtifactStore(tmp_path / "canonical")

    registry_receipt = store.write(Path("method_registry.json"), registry)
    first = store.write(Path("fairness_certificate.json"), certificate)
    second = store.write(Path("fairness_certificate.json"), certificate)
    restored = store.read(
        FairnessCertificate,
        ArtifactReadRequest(Path("fairness_certificate.json"), second.sha256),
    )

    assert certificate.registry_sha256 == registry_receipt.sha256
    assert first.sha256 == second.sha256
    assert restored == certificate
