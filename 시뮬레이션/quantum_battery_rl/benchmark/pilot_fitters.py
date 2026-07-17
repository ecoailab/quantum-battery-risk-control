"""Budget-specific fitter construction for the convergence pilot."""

from __future__ import annotations

from pathlib import Path

from .controller import ValidationDraws
from .cvar_control import CvarControlConfig, CvarControlFitter
from .manifest import CanonicalManifest
from .mean_control import MeanControlConfig, MeanControlFitter
from .nominal_control import NominalControlConfig, NominalControlFitter
from .ppo_control import PpoControlConfig, PpoControlFitter
from .smoke_fitters import SmokeFitterSpec, source_hash


FINITE_DIFFERENCE_METHODS = (
    "nominal-lbfgsb-finite-difference",
    "saa-mean-lbfgsb-finite-difference",
    "cvar-0.1-lbfgsb-finite-difference",
)
PPO_METHOD = "ppo-time-only-domain-randomized"
PILOT_METHODS = (*FINITE_DIFFERENCE_METHODS, PPO_METHOD)


def build_pilot_fitter(
    project_root: Path,
    manifest: CanonicalManifest,
    manifest_sha256: str,
    validation_draws: ValidationDraws,
    method_id: str,
    budget: int,
) -> SmokeFitterSpec:
    physics = manifest.physics
    if method_id == FINITE_DIFFERENCE_METHODS[0]:
        implementation_hash = source_hash(
            project_root,
            (Path("quantum_battery_rl/benchmark/nominal_control.py"),),
        )
        fitter = NominalControlFitter(
            NominalControlConfig(
                physics=physics,
                manifest_sha256=manifest_sha256,
                implementation_sha256=implementation_hash,
                max_iterations=budget,
                max_objective_evaluations=budget,
            ),
        )
    elif method_id == FINITE_DIFFERENCE_METHODS[1]:
        implementation_hash = source_hash(
            project_root,
            (Path("quantum_battery_rl/benchmark/mean_control.py"),),
        )
        fitter = MeanControlFitter(
            MeanControlConfig(
                physics=physics,
                manifest_sha256=manifest_sha256,
                implementation_sha256=implementation_hash,
                max_iterations=budget,
                max_objective_evaluations=budget,
            ),
        )
    elif method_id == FINITE_DIFFERENCE_METHODS[2]:
        implementation_hash = source_hash(
            project_root,
            (Path("quantum_battery_rl/benchmark/cvar_control.py"),),
        )
        fitter = CvarControlFitter(
            CvarControlConfig(
                physics=physics,
                manifest_sha256=manifest_sha256,
                implementation_sha256=implementation_hash,
                max_iterations=budget,
                max_objective_evaluations=budget,
            ),
        )
    elif method_id == PPO_METHOD:
        implementation_hash = source_hash(
            project_root,
            (
                Path("quantum_battery_rl/benchmark/pilot_fitters.py"),
                Path("quantum_battery_rl/benchmark/ppo_control.py"),
                Path("quantum_battery_rl/agents/ppo.py"),
            ),
        )
        fitter = PpoControlFitter(
            PpoControlConfig(
                physics=physics,
                manifest_sha256=manifest_sha256,
                implementation_sha256=implementation_hash,
                environment_steps=budget,
                checkpoint_interval_steps=(
                    manifest.optimization.ppo_environment_step_stages[0]
                ),
                learning_rate=0.001,
            ),
            validation_draws,
        )
    else:
        raise ValueError(f"unsupported pilot method: {method_id}")
    return SmokeFitterSpec(method_id, implementation_hash, fitter)
