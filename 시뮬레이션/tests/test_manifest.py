from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from quantum_battery_rl.benchmark.manifest import load_manifest


VALID_MANIFEST: Final = """{
  "schema_version": "1.0",
  "physics": {
    "t1": 100.0,
    "t2": 80.0,
    "omega_q": 5.0,
    "max_omega": 0.25,
    "n_steps": 100,
    "dt": 0.1
  },
  "uncertainty": {
    "source": "synthetic_joint_stress_test",
    "severity_fractions": [0.1, 0.3, 0.5],
    "differential_ratio": 0.5
  },
  "splits": {
    "seeds": {"train": 1101, "validation": 2202, "test": 3303},
    "scenarios": {"train": 128, "validation": 128, "test": 256}
  },
  "optimization": {
    "optimizer_seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    "grape_evaluation_stages": [500, 1000, 2000],
    "ppo_environment_step_stages": [50000, 100000, 200000],
    "smoke_scenarios": 4
  },
  "statistics": {
    "cvar_alpha": 0.1,
    "bootstrap_samples": 2000,
    "q_threshold": 0.05,
    "cliff_threshold": 0.33,
    "confidence_level": 0.95
  },
  "output": {"root": "results/canonical"}
}"""


def _write_manifest(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_manifest_is_parsed_and_frozen(tmp_path: Path) -> None:
    # Given
    path = _write_manifest(tmp_path, VALID_MANIFEST)

    # When
    manifest = load_manifest(path)

    # Then
    assert manifest.schema_version == "1.0"
    assert manifest.splits.scenarios.test == 256
    with pytest.raises(ValidationError):
        setattr(manifest, "schema_version", "2.0")


@pytest.mark.parametrize(
    "invalid_manifest",
    [
        VALID_MANIFEST.replace(
            '"schema_version": "1.0",',
            '"schema_version": "1.0", "unknown": true,',
        ),
        VALID_MANIFEST.replace(
            ',\n  "output": {"root": "results/canonical"}',
            "",
        ),
        VALID_MANIFEST.replace('"validation": 2202', '"validation": 1101'),
        VALID_MANIFEST.replace('"t2": 80.0', '"t2": 201.0'),
        VALID_MANIFEST.replace('"results/canonical"', '"results/legacy"'),
        VALID_MANIFEST.replace('"n_steps": 100', '"n_steps": "100"'),
    ],
)
def test_invalid_manifest_is_rejected(
    tmp_path: Path,
    invalid_manifest: str,
) -> None:
    # Given
    path = _write_manifest(tmp_path, invalid_manifest)

    # When / Then
    with pytest.raises(ValidationError):
        _ = load_manifest(path)
