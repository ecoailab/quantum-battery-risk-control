from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from pydantic import ValidationError

from quantum_battery_rl.benchmark import (
    ControllerFitter,
    FitCost,
    FitMetadata,
    FitProvenance,
    FittedController,
    InvalidProvenanceHashError,
    NonTrainingDrawSetError,
    NonValidationDrawSetError,
    ObjectiveContribution,
    PulseController,
    PulseHorizonMismatchError,
    TrainingDraws,
    ValidationDraws,
)
from quantum_battery_rl.benchmark.uncertainty import DrawSet


DRAW_ROOT: Final = Path(__file__).parents[1] / "results" / "canonical" / "draws"
MANIFEST_HASH: Final = "a0673ec163fbc5bf1617e59ee8cc0e87d81e60f9994e91542206106b89259464"
IMPLEMENTATION_HASH: Final = "1" * 64


def _load_draw_set(split: str) -> tuple[DrawSet, str]:
    path = DRAW_ROOT / f"{split}.json"
    return (
        DrawSet.model_validate_json(path.read_text(encoding="utf-8")),
        sha256(path.read_bytes()).hexdigest(),
    )


def _metadata() -> FitMetadata:
    return FitMetadata(
        method_id="test-controller",
        converged=True,
        termination_reason="test fixture",
        cost=FitCost(
            objective_evaluations=3,
            gradient_evaluations=1,
            environment_steps=300,
            wall_time_seconds=0.25,
        ),
        provenance=FitProvenance(
            manifest_sha256=MANIFEST_HASH,
            training_draws_sha256="0" * 64,
            validation_draws_sha256=None,
            implementation_sha256=IMPLEMENTATION_HASH,
            optimizer_seed=42,
        ),
        objective_contributions=(
            ObjectiveContribution(draw_id="fixture", value=0.5),
        ),
    )


def _fitted_controller() -> FittedController:
    return FittedController(
        schema_version="1.0",
        n_steps=2,
        max_action=0.25,
        controls=((0.25, 0.0), (0.0, -0.25)),
        metadata=_metadata(),
    )


def test_training_draws_reject_validation_test_and_bad_hashes() -> None:
    # Given
    train_set, train_hash = _load_draw_set("train")
    validation_set, validation_hash = _load_draw_set("validation")
    test_set, test_hash = _load_draw_set("test")

    # When
    training_draws = TrainingDraws(train_set, train_hash)

    # Then
    assert training_draws.draw_set.split == "train"
    with pytest.raises(NonTrainingDrawSetError):
        _ = TrainingDraws(validation_set, validation_hash)
    with pytest.raises(NonTrainingDrawSetError):
        _ = TrainingDraws(test_set, test_hash)
    with pytest.raises(InvalidProvenanceHashError):
        _ = TrainingDraws(train_set, "not-a-sha256")


def test_fitted_pulse_is_read_only_and_horizon_checked() -> None:
    # Given
    fitted = _fitted_controller()

    # When
    pulse = fitted.pulse(2)

    # Then
    assert pulse.shape == (2, 2)
    assert pulse.dtype == np.float64
    assert not pulse.flags.writeable
    with pytest.raises(ValueError):
        pulse[0, 0] = 0.0
    with pytest.raises(PulseHorizonMismatchError):
        _ = fitted.pulse(3)


def test_fit_metadata_is_complete_strict_and_frozen() -> None:
    # Given / When
    metadata = _metadata()

    # Then
    assert set(metadata.cost.model_dump()) == {
        "objective_evaluations",
        "gradient_evaluations",
        "environment_steps",
        "wall_time_seconds",
    }
    assert set(metadata.provenance.model_dump()) == {
        "manifest_sha256",
        "training_draws_sha256",
        "validation_draws_sha256",
        "implementation_sha256",
        "optimizer_seed",
    }
    with pytest.raises(ValidationError):
        setattr(metadata, "converged", False)
    with pytest.raises(ValidationError):
        _ = FitCost.model_validate(
            {
                "objective_evaluations": 1,
                "gradient_evaluations": 0,
                "environment_steps": 100,
            },
        )


def test_validation_draws_reject_train_test_and_bad_hashes() -> None:
    # Given
    train_set, train_hash = _load_draw_set("train")
    validation_set, validation_hash = _load_draw_set("validation")
    test_set, test_hash = _load_draw_set("test")

    # When
    validation_draws = ValidationDraws(validation_set, validation_hash)

    # Then
    assert validation_draws.draw_set.split == "validation"
    with pytest.raises(NonValidationDrawSetError):
        _ = ValidationDraws(train_set, train_hash)
    with pytest.raises(NonValidationDrawSetError):
        _ = ValidationDraws(test_set, test_hash)
    with pytest.raises(InvalidProvenanceHashError):
        _ = ValidationDraws(validation_set, "not-a-sha256")


@pytest.mark.parametrize(
    "controls",
    [
        ((0.1, 0.0),),
        ((0.3, 0.0), (0.0, 0.0)),
        ((float("nan"), 0.0), (0.0, 0.0)),
    ],
)
def test_fitted_controller_rejects_invalid_pulses(
    controls: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(ValidationError):
        _ = FittedController(
            schema_version="1.0",
            n_steps=2,
            max_action=0.25,
            controls=controls,
            metadata=_metadata(),
        )


class _FixtureFitter:
    _fitted: FittedController

    def __init__(self, fitted: FittedController) -> None:
        self._fitted = fitted

    def fit(self, training_draws: TrainingDraws, seed: int) -> FittedController:
        assert training_draws.draw_set.split == "train"
        assert seed == self._fitted.metadata.provenance.optimizer_seed
        return self._fitted


def test_structural_fit_and_pulse_protocols_share_common_outputs() -> None:
    # Given
    draw_set, draw_hash = _load_draw_set("train")
    training_draws = TrainingDraws(draw_set, draw_hash)
    fitted = _fitted_controller()
    fitter: ControllerFitter = _FixtureFitter(fitted)
    pulse_controller: PulseController = fitted

    # When
    actual = fitter.fit(training_draws, seed=42)

    # Then
    assert isinstance(fitter, ControllerFitter)
    assert isinstance(pulse_controller, PulseController)
    assert actual is fitted
    assert set(FittedController.model_fields) == {
        "schema_version",
        "n_steps",
        "max_action",
        "controls",
        "metadata",
    }
