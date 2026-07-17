from __future__ import annotations

from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Final

from quantum_battery_rl.benchmark.manifest import load_manifest
from quantum_battery_rl.benchmark.uncertainty import (
    generate_draw_sets,
    write_draw_set,
)


MANIFEST_PATH: Final = Path(__file__).parents[1] / "canonical_manifest.json"
MANIFEST_HASH: Final = "a0673ec163fbc5bf1617e59ee8cc0e87d81e60f9994e91542206106b89259464"


def test_draw_generation_is_deterministic_and_disjoint() -> None:
    # Given
    manifest = load_manifest(MANIFEST_PATH)

    # When
    first = generate_draw_sets(manifest, MANIFEST_HASH)
    second = generate_draw_sets(manifest, MANIFEST_HASH)

    # Then
    assert first == second
    identifier_sets = [{draw.draw_id for draw in draw_set.draws} for draw_set in first]
    assert identifier_sets[0].isdisjoint(identifier_sets[1])
    assert identifier_sets[0].isdisjoint(identifier_sets[2])
    assert identifier_sets[1].isdisjoint(identifier_sets[2])
    parameter_sets = [
        {(draw.severity_fraction, draw.t1, draw.t2) for draw in draw_set.draws}
        for draw_set in first
    ]
    assert parameter_sets[0].isdisjoint(parameter_sets[1])
    assert parameter_sets[0].isdisjoint(parameter_sets[2])
    assert parameter_sets[1].isdisjoint(parameter_sets[2])


def test_source_manifest_hash_is_current() -> None:
    assert sha256(MANIFEST_PATH.read_bytes()).hexdigest() == MANIFEST_HASH


def test_draw_counts_match_every_split_and_severity_band() -> None:
    # Given
    manifest = load_manifest(MANIFEST_PATH)

    # When
    draw_sets = generate_draw_sets(manifest, MANIFEST_HASH)

    # Then
    expected_counts = (
        manifest.splits.scenarios.train * len(manifest.uncertainty.severity_fractions),
        manifest.splits.scenarios.validation * len(manifest.uncertainty.severity_fractions),
        manifest.splits.scenarios.test * len(manifest.uncertainty.severity_fractions),
    )
    assert tuple(len(draw_set.draws) for draw_set in draw_sets) == expected_counts


def test_draws_stay_inside_declared_bounds_and_physical_region() -> None:
    # Given
    manifest = load_manifest(MANIFEST_PATH)

    # When
    draw_sets = generate_draw_sets(manifest, MANIFEST_HASH)

    # Then
    for draw_set in draw_sets:
        for draw in draw_set.draws:
            t1_factor = draw.t1 / manifest.physics.t1
            t2_factor = draw.t2 / manifest.physics.t2
            lower = 1.0 - draw.severity_fraction
            upper = 1.0 + draw.severity_fraction
            assert lower <= t1_factor <= upper
            assert lower <= t2_factor <= upper
            assert draw.t2 <= 2.0 * draw.t1


def test_physicality_over_ten_thousand_draws() -> None:
    # Given
    manifest = load_manifest(MANIFEST_PATH)
    large_scenarios = manifest.splits.scenarios.model_copy(
        update={"train": 3_334, "validation": 1, "test": 1},
    )
    large_splits = manifest.splits.model_copy(
        update={"scenarios": large_scenarios},
    )
    large_manifest = manifest.model_copy(update={"splits": large_splits})

    # When
    train_draws, _, _ = generate_draw_sets(large_manifest, MANIFEST_HASH)

    # Then
    assert len(train_draws.draws) == 10_002
    assert all(draw.t2 <= 2.0 * draw.t1 for draw in train_draws.draws)


def test_persisted_draw_set_is_byte_identical_on_repeat(tmp_path: Path) -> None:
    # Given
    manifest = load_manifest(MANIFEST_PATH)
    train_draws, _, _ = generate_draw_sets(manifest, MANIFEST_HASH)

    # When
    first_path = write_draw_set(train_draws, tmp_path)
    first_bytes = first_path.read_bytes()
    second_path = write_draw_set(train_draws, tmp_path)

    # Then
    assert first_path == second_path
    assert first_bytes == second_path.read_bytes()


def test_joint_modes_are_not_independent_parameter_draws() -> None:
    # Given
    manifest = load_manifest(MANIFEST_PATH)
    train_draws, _, _ = generate_draw_sets(manifest, MANIFEST_HASH)

    # When
    t1_factors = tuple(draw.t1 / manifest.physics.t1 for draw in train_draws.draws)
    t2_factors = tuple(draw.t2 / manifest.physics.t2 for draw in train_draws.draws)
    t1_mean = sum(t1_factors) / len(t1_factors)
    t2_mean = sum(t2_factors) / len(t2_factors)
    covariance = sum(
        (t1 - t1_mean) * (t2 - t2_mean)
        for t1, t2 in zip(t1_factors, t2_factors, strict=True)
    )
    t1_variance = sum((value - t1_mean) ** 2 for value in t1_factors)
    t2_variance = sum((value - t2_mean) ** 2 for value in t2_factors)
    correlation = covariance / sqrt(t1_variance * t2_variance)

    # Then
    assert abs(correlation) > 0.1
