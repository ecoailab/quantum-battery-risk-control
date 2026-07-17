from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from shutil import copy2
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_store import ArtifactReadRequest, ArtifactStore
from quantum_battery_rl.benchmark.full_fit_models import (
    FullFitArtifact,
    FullFitConfigArtifact,
)
from quantum_battery_rl.benchmark.full_fit_runner import (
    FullFitRequest,
    MissingFullFitConfigError,
    StaleFullFitInputError,
    plan_full_fits,
    run_full_fits,
)


PROJECT_ROOT: Final = Path(__file__).parents[1]
CANONICAL_SOURCE: Final = PROJECT_ROOT / "results" / "canonical"
MANIFEST_PATH: Final = PROJECT_ROOT / "canonical_manifest.json"


def _copy_inputs(tmp_path: Path) -> Path:
    canonical_root = tmp_path / "canonical"
    for relative in (
        Path("method_registry.json"),
        Path("fairness_certificate.json"),
        Path("draws/train.json"),
        Path("draws/validation.json"),
        Path("pilot/budget_amendment.json"),
    ):
        destination = canonical_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = copy2(CANONICAL_SOURCE / relative, destination)
    return canonical_root


def _write_config(canonical_root: Path) -> None:
    manifest_hash = sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    fairness_hash = sha256(
        (canonical_root / "fairness_certificate.json").read_bytes(),
    ).hexdigest()
    amendment_hash = sha256(
        (canonical_root / "pilot" / "budget_amendment.json").read_bytes(),
    ).hexdigest()
    config = FullFitConfigArtifact(
        schema_version="1.0",
        artifact_type="full_fit_config",
        manifest_sha256=manifest_hash,
        fairness_certificate_sha256=fairness_hash,
        budget_amendment_sha256=amendment_hash,
        bang_bang_axis_count=2,
        sinusoidal_frequency_count=2,
        sinusoidal_phase_count=2,
        test_data_accessed=False,
    )
    _ = ArtifactStore(canonical_root).write(Path("full_fit_config.json"), config)


def test_full_fit_plan_contains_every_certified_method_seed_and_budget(
    tmp_path: Path,
) -> None:
    # Given
    canonical_root = _copy_inputs(tmp_path)
    _write_config(canonical_root)
    request = FullFitRequest(PROJECT_ROOT, MANIFEST_PATH, canonical_root)

    # When
    plan = plan_full_fits(request)

    # Then
    assert len(plan.identities) == 70
    assert len({(item.method_id, item.optimizer_seed) for item in plan.identities}) == 70
    assert {item.optimizer_seed for item in plan.identities} == set(range(42, 52))
    selected = {(item.method_id, item.requested_budget) for item in plan.identities}
    assert ("nominal-lbfgsb-finite-difference", 32000) in selected
    assert ("saa-mean-lbfgsb-finite-difference", 32000) in selected
    assert ("cvar-0.1-lbfgsb-finite-difference", 32000) in selected
    assert ("ppo-time-only-domain-randomized", 200000) in selected
    assert ("bang-bang-ensemble-grid", 200) in selected
    assert ("sinusoidal-ensemble-grid", 4) in selected
    assert ("random-seeded-reference", 1) in selected


def test_full_fit_runner_is_seed_shardable_and_resume_safe(tmp_path: Path) -> None:
    # Given
    canonical_root = _copy_inputs(tmp_path)
    _write_config(canonical_root)
    request = FullFitRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        canonical_root,
        method_ids=("random-seeded-reference",),
        optimizer_seeds=(42, 43),
    )

    # When
    first = run_full_fits(request)
    second = run_full_fits(request)

    # Then
    assert first.reused_fit_count == 0
    assert second.reused_fit_count == 2
    assert second.fit_hashes == first.fit_hashes
    assert len(first.fit_hashes) == 2
    assert not (canonical_root / "draws" / "test.json").exists()
    assert (canonical_root / "fits" / "random-seeded-reference" / "42.json").is_file()
    assert (canonical_root / "fits" / "random-seeded-reference" / "43.json").is_file()


def test_full_fit_runner_fails_closed_without_predeclared_grid_config(
    tmp_path: Path,
) -> None:
    # Given
    canonical_root = _copy_inputs(tmp_path)
    request = FullFitRequest(PROJECT_ROOT, MANIFEST_PATH, canonical_root)

    # When / Then
    with pytest.raises(MissingFullFitConfigError):
        _ = plan_full_fits(request)


def test_full_fit_resume_rejects_hash_linked_but_stale_artifact(
    tmp_path: Path,
) -> None:
    # Given
    canonical_root = _copy_inputs(tmp_path)
    _write_config(canonical_root)
    request = FullFitRequest(
        PROJECT_ROOT,
        MANIFEST_PATH,
        canonical_root,
        method_ids=("random-seeded-reference",),
        optimizer_seeds=(42,),
    )
    report = run_full_fits(request)
    method_id, seed, artifact_hash = report.fit_hashes[0]
    relative = Path(method_id) / f"{seed}.json"
    store = ArtifactStore(canonical_root / "fits")
    artifact = store.read(
        FullFitArtifact,
        ArtifactReadRequest(relative, artifact_hash),
    )
    stale = artifact.model_copy(update={"full_fit_config_sha256": "e" * 64})
    _ = store.write(relative, stale)

    # When / Then
    with pytest.raises(StaleFullFitInputError, match="stale full fit"):
        _ = run_full_fits(request)
