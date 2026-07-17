"""Reduced-budget fitter construction for the canonical smoke run."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .controller import ControllerFitter, ValidationDraws
from .cvar_control import CvarControlConfig, CvarControlFitter
from .heuristic_control import (
    BangBangFitter,
    HeuristicControlConfig,
    RandomReferenceFitter,
    SinusoidalFitter,
)
from .manifest import CanonicalManifest
from .mean_control import MeanControlConfig, MeanControlFitter
from .nominal_control import NominalControlConfig, NominalControlFitter
from .ppo_control import PpoControlConfig, PpoControlFitter


@dataclass(frozen=True, slots=True)
class SmokeFitterSpec:
    method_id: str
    implementation_sha256: str
    fitter: ControllerFitter


def build_smoke_fitter_specs(
    project_root: Path,
    manifest: CanonicalManifest,
    manifest_sha256: str,
    validation_draws: ValidationDraws,
) -> tuple[SmokeFitterSpec, ...]:
    physics = manifest.physics
    finite_difference_budget = max(8, 2 * physics.n_steps + 2)
    nominal_hash = source_hash(
        project_root,
        (Path("quantum_battery_rl/benchmark/nominal_control.py"),),
    )
    mean_hash = source_hash(
        project_root,
        (Path("quantum_battery_rl/benchmark/mean_control.py"),),
    )
    cvar_hash = source_hash(
        project_root,
        (Path("quantum_battery_rl/benchmark/cvar_control.py"),),
    )
    ppo_hash = source_hash(
        project_root,
        (
            Path("quantum_battery_rl/benchmark/ppo_control.py"),
            Path("quantum_battery_rl/agents/ppo.py"),
        ),
    )
    heuristic_hash = source_hash(
        project_root,
        (Path("quantum_battery_rl/benchmark/heuristic_control.py"),),
    )
    heuristic_config = HeuristicControlConfig(
        physics=physics,
        manifest_sha256=manifest_sha256,
        implementation_sha256=heuristic_hash,
        axis_count=1,
        frequency_count=1,
        phase_count=1,
    )
    return (
        SmokeFitterSpec(
            "nominal-lbfgsb-finite-difference",
            nominal_hash,
            NominalControlFitter(
                NominalControlConfig(
                    physics=physics,
                    manifest_sha256=manifest_sha256,
                    implementation_sha256=nominal_hash,
                    max_iterations=1,
                    max_objective_evaluations=finite_difference_budget,
                ),
            ),
        ),
        SmokeFitterSpec(
            "saa-mean-lbfgsb-finite-difference",
            mean_hash,
            MeanControlFitter(
                MeanControlConfig(
                    physics=physics,
                    manifest_sha256=manifest_sha256,
                    implementation_sha256=mean_hash,
                    max_iterations=1,
                    max_objective_evaluations=finite_difference_budget,
                ),
            ),
        ),
        SmokeFitterSpec(
            "cvar-0.1-lbfgsb-finite-difference",
            cvar_hash,
            CvarControlFitter(
                CvarControlConfig(
                    physics=physics,
                    manifest_sha256=manifest_sha256,
                    implementation_sha256=cvar_hash,
                    max_iterations=1,
                    max_objective_evaluations=finite_difference_budget,
                ),
            ),
        ),
        SmokeFitterSpec(
            "ppo-time-only-domain-randomized",
            ppo_hash,
            PpoControlFitter(
                PpoControlConfig(
                    physics=physics,
                    manifest_sha256=manifest_sha256,
                    implementation_sha256=ppo_hash,
                    environment_steps=physics.n_steps,
                    checkpoint_interval_steps=physics.n_steps,
                    learning_rate=0.001,
                ),
                validation_draws,
            ),
        ),
        SmokeFitterSpec(
            "bang-bang-ensemble-grid",
            heuristic_hash,
            BangBangFitter(heuristic_config),
        ),
        SmokeFitterSpec(
            "sinusoidal-ensemble-grid",
            heuristic_hash,
            SinusoidalFitter(heuristic_config),
        ),
        SmokeFitterSpec(
            "random-seeded-reference",
            heuristic_hash,
            RandomReferenceFitter(heuristic_config),
        ),
    )


def source_hash(project_root: Path, relative_paths: tuple[Path, ...]) -> str:
    digest = sha256()
    for relative_path in relative_paths:
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((project_root / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
