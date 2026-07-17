"""Construct canonical full-budget fitters from a predeclared fit identity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from .controller import ControllerFitter, ValidationDraws
from .cvar_control import CvarControlConfig, CvarControlFitter
from .full_fit_models import FullFitConfigArtifact, FullFitIdentity, FullMethod
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
from .smoke_fitters import source_hash


@dataclass(frozen=True, slots=True)
class FullFitterContext:
    project_root: Path
    manifest: CanonicalManifest
    manifest_sha256: str
    validation_draws: ValidationDraws
    config: FullFitConfigArtifact


@dataclass(frozen=True, slots=True)
class FullFitterSpec:
    identity: FullFitIdentity
    implementation_sha256: str
    fitter: ControllerFitter


def build_full_fitter(
    context: FullFitterContext,
    identity: FullFitIdentity,
) -> FullFitterSpec:
    return _BUILDERS[identity.method_id](context, identity)


def _nominal(
    context: FullFitterContext,
    identity: FullFitIdentity,
) -> FullFitterSpec:
    physics = context.manifest.physics
    implementation_hash = source_hash(
        context.project_root,
        (
            Path("quantum_battery_rl/benchmark/nominal_control.py"),
            Path("quantum_battery_rl/env/quantum_dynamics.py"),
        ),
    )
    fitter: ControllerFitter = NominalControlFitter(
        NominalControlConfig(
            physics=physics,
            manifest_sha256=context.manifest_sha256,
            implementation_sha256=implementation_hash,
            max_iterations=identity.requested_budget,
            max_objective_evaluations=identity.requested_budget,
        ),
    )
    return FullFitterSpec(identity, implementation_hash, fitter)


def _mean(
    context: FullFitterContext,
    identity: FullFitIdentity,
) -> FullFitterSpec:
    implementation_hash = _ensemble_hash(context, "mean_control.py")
    fitter: ControllerFitter = MeanControlFitter(
        MeanControlConfig(
            physics=context.manifest.physics,
            manifest_sha256=context.manifest_sha256,
            implementation_sha256=implementation_hash,
            max_iterations=identity.requested_budget,
            max_objective_evaluations=identity.requested_budget,
        ),
    )
    return FullFitterSpec(identity, implementation_hash, fitter)


def _cvar(
    context: FullFitterContext,
    identity: FullFitIdentity,
) -> FullFitterSpec:
    implementation_hash = _ensemble_hash(context, "cvar_control.py")
    fitter: ControllerFitter = CvarControlFitter(
        CvarControlConfig(
            physics=context.manifest.physics,
            manifest_sha256=context.manifest_sha256,
            implementation_sha256=implementation_hash,
            max_iterations=identity.requested_budget,
            max_objective_evaluations=identity.requested_budget,
        ),
    )
    return FullFitterSpec(identity, implementation_hash, fitter)


def _ppo(
    context: FullFitterContext,
    identity: FullFitIdentity,
) -> FullFitterSpec:
    implementation_hash = source_hash(
        context.project_root,
        (
            Path("quantum_battery_rl/benchmark/ppo_control.py"),
            Path("quantum_battery_rl/benchmark/environment.py"),
            Path("quantum_battery_rl/benchmark/batch_dynamics.py"),
            Path("quantum_battery_rl/agents/ppo.py"),
            Path("quantum_battery_rl/env/quantum_dynamics.py"),
        ),
    )
    fitter: ControllerFitter = PpoControlFitter(
        PpoControlConfig(
            physics=context.manifest.physics,
            manifest_sha256=context.manifest_sha256,
            implementation_sha256=implementation_hash,
            environment_steps=identity.requested_budget,
            checkpoint_interval_steps=(
                context.manifest.optimization.ppo_environment_step_stages[0]
            ),
            learning_rate=0.001,
        ),
        context.validation_draws,
    )
    return FullFitterSpec(identity, implementation_hash, fitter)


HeuristicBuilder = Callable[[HeuristicControlConfig], ControllerFitter]
_HEURISTIC_BUILDERS: Final[Mapping[FullMethod, HeuristicBuilder]] = MappingProxyType(
    {
        FullMethod.BANG_BANG: BangBangFitter,
        FullMethod.SINUSOIDAL: SinusoidalFitter,
        FullMethod.RANDOM: RandomReferenceFitter,
    },
)


def _heuristic(
    context: FullFitterContext,
    identity: FullFitIdentity,
) -> FullFitterSpec:
    implementation_hash = _ensemble_hash(context, "heuristic_control.py")
    config = HeuristicControlConfig(
        physics=context.manifest.physics,
        manifest_sha256=context.manifest_sha256,
        implementation_sha256=implementation_hash,
        axis_count=context.config.bang_bang_axis_count,
        frequency_count=context.config.sinusoidal_frequency_count,
        phase_count=context.config.sinusoidal_phase_count,
    )
    fitter = _HEURISTIC_BUILDERS[identity.method_id](config)
    return FullFitterSpec(identity, implementation_hash, fitter)


def _ensemble_hash(context: FullFitterContext, controller_file: str) -> str:
    return source_hash(
        context.project_root,
        (
            Path("quantum_battery_rl/benchmark") / controller_file,
            Path("quantum_battery_rl/benchmark/batch_dynamics.py"),
            Path("quantum_battery_rl/env/quantum_dynamics.py"),
        ),
    )


FullBuilder = Callable[[FullFitterContext, FullFitIdentity], FullFitterSpec]
_BUILDERS: Final[Mapping[FullMethod, FullBuilder]] = MappingProxyType(
    {
        FullMethod.NOMINAL: _nominal,
        FullMethod.MEAN: _mean,
        FullMethod.CVAR: _cvar,
        FullMethod.PPO: _ppo,
        FullMethod.BANG_BANG: _heuristic,
        FullMethod.SINUSOIDAL: _heuristic,
        FullMethod.RANDOM: _heuristic,
    },
)
