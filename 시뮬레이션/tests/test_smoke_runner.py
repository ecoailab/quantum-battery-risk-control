from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

from quantum_battery_rl.benchmark.artifact_models import FitArtifact, RawResultsArtifact
from quantum_battery_rl.benchmark.artifact_store import ArtifactReadRequest, ArtifactStore
from quantum_battery_rl.benchmark.fairness import (
    audit_registry,
    build_default_registry,
    canonical_simulator_sha256,
)
from quantum_battery_rl.benchmark.manifest import PhysicsConfig, load_manifest
from quantum_battery_rl.benchmark.smoke_runner import run_smoke
from quantum_battery_rl.benchmark.smoke_artifacts import StaleSmokeArtifactError
from quantum_battery_rl.benchmark.uncertainty import DrawSet


PROJECT_ROOT: Final = Path(__file__).parents[1]
DRAW_ROOT: Final = PROJECT_ROOT / "results" / "canonical" / "draws"
EXPECTED_METHODS: Final = {
    "nominal-lbfgsb-finite-difference",
    "saa-mean-lbfgsb-finite-difference",
    "cvar-0.1-lbfgsb-finite-difference",
    "ppo-time-only-domain-randomized",
    "bang-bang-ensemble-grid",
    "sinusoidal-ensemble-grid",
    "random-seeded-reference",
}


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = load_manifest(PROJECT_ROOT / "canonical_manifest.json").model_copy(
        update={
            "physics": PhysicsConfig(
                t1=100.0,
                t2=80.0,
                omega_q=5.0,
                max_omega=0.25,
                n_steps=2,
                dt=0.1,
            ),
        },
    )
    manifest_path = tmp_path / "canonical_manifest.json"
    _ = manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    canonical_root = tmp_path / "canonical"
    draw_root = canonical_root / "draws"
    draw_root.mkdir(parents=True)
    for split in ("train", "validation"):
        source = DrawSet.model_validate_json((DRAW_ROOT / f"{split}.json").read_bytes())
        draw_set = source.model_copy(
            update={"source_manifest_sha256": manifest_hash},
        )
        _ = (draw_root / f"{split}.json").write_text(
            draw_set.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    registry = build_default_registry(
        manifest,
        manifest_hash,
        canonical_simulator_sha256(PROJECT_ROOT),
    )
    certificate = audit_registry(registry, manifest)
    store = ArtifactStore(canonical_root)
    _ = store.write(Path("method_registry.json"), registry)
    _ = store.write(Path("fairness_certificate.json"), certificate)
    return manifest_path, canonical_root


def test_smoke_run_fits_and_validates_all_methods_without_test_draws(
    tmp_path: Path,
) -> None:
    manifest_path, canonical_root = _prepare_inputs(tmp_path)

    report = run_smoke(PROJECT_ROOT, manifest_path, canonical_root)

    assert set(report.fit_sha256s) == EXPECTED_METHODS
    assert report.reused_fit_count == 0
    assert not report.raw_results_reused
    assert not (canonical_root / "draws" / "test.json").exists()
    assert len(tuple((canonical_root / "smoke" / "fits").glob("*.json"))) == 7
    raw = ArtifactStore(canonical_root / "smoke").read(
        RawResultsArtifact,
        ArtifactReadRequest(Path("raw_results.json"), report.raw_results_sha256),
    )
    assert raw.evaluation_split == "validation"
    assert len(raw.records) == 7 * 4
    assert {record.method_id for record in raw.records} == EXPECTED_METHODS


def test_smoke_resume_reuses_exact_schema_valid_artifacts(tmp_path: Path) -> None:
    manifest_path, canonical_root = _prepare_inputs(tmp_path)
    first = run_smoke(PROJECT_ROOT, manifest_path, canonical_root)

    second = run_smoke(PROJECT_ROOT, manifest_path, canonical_root)

    assert second.fit_sha256s == first.fit_sha256s
    assert second.raw_results_sha256 == first.raw_results_sha256
    assert second.reused_fit_count == 7
    assert second.raw_results_reused
    store = ArtifactStore(canonical_root / "smoke")
    for method_id, artifact_hash in second.fit_sha256s.items():
        artifact = store.read(
            FitArtifact,
            ArtifactReadRequest(Path("fits") / f"{method_id}.json", artifact_hash),
        )
        assert artifact.controller.metadata.method_id == method_id


def test_smoke_resume_rejects_valid_but_stale_fit(tmp_path: Path) -> None:
    manifest_path, canonical_root = _prepare_inputs(tmp_path)
    report = run_smoke(PROJECT_ROOT, manifest_path, canonical_root)
    method_id = "nominal-lbfgsb-finite-difference"
    relative = Path("fits") / f"{method_id}.json"
    store = ArtifactStore(canonical_root / "smoke")
    artifact = store.read(
        FitArtifact,
        ArtifactReadRequest(relative, report.fit_sha256s[method_id]),
    )
    stale_hash = "e" * 64
    provenance = artifact.controller.metadata.provenance.model_copy(
        update={"implementation_sha256": stale_hash},
    )
    metadata = artifact.controller.metadata.model_copy(update={"provenance": provenance})
    controller = artifact.controller.model_copy(update={"metadata": metadata})
    stale = FitArtifact(
        schema_version="1.0",
        artifact_type="fit",
        manifest_sha256=artifact.manifest_sha256,
        training_draws_sha256=artifact.training_draws_sha256,
        implementation_sha256=stale_hash,
        controller=controller,
    )
    _ = store.write(relative, stale)

    with pytest.raises(StaleSmokeArtifactError, match="stale smoke fit"):
        _ = run_smoke(PROJECT_ROOT, manifest_path, canonical_root)
