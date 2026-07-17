"""Canonical benchmark contracts."""

from .controller import (
    ControllerFitter as ControllerFitter,
    FitCost as FitCost,
    FitMetadata as FitMetadata,
    FitProvenance as FitProvenance,
    FittedController as FittedController,
    InvalidProvenanceHashError as InvalidProvenanceHashError,
    NonTrainingDrawSetError as NonTrainingDrawSetError,
    NonValidationDrawSetError as NonValidationDrawSetError,
    ObjectiveContribution as ObjectiveContribution,
    PulseController as PulseController,
    PulseHorizonMismatchError as PulseHorizonMismatchError,
    TrainingDraws as TrainingDraws,
    ValidationDraws as ValidationDraws,
)
from .budget_models import (
    BudgetAmendmentArtifact as BudgetAmendmentArtifact,
    BudgetDecision as BudgetDecision,
    PilotStageResult as PilotStageResult,
    PlateauCriterion as PlateauCriterion,
    assess_budget as assess_budget,
)
from .convergence_pilot import run_convergence_pilot as run_convergence_pilot
from .cvar_control import (
    CvarControlConfig as CvarControlConfig,
    CvarControlFitter as CvarControlFitter,
    select_lower_tail as select_lower_tail,
)
from .environment import (
    CanonicalOpenLoopEnv as CanonicalOpenLoopEnv,
    EpisodeParameters as EpisodeParameters,
)
from .fairness import (
    DuplicateMethodRegistrationError as DuplicateMethodRegistrationError,
    FairnessCertificate as FairnessCertificate,
    FairnessCheck as FairnessCheck,
    MethodRegistration as MethodRegistration,
    MethodRegistry as MethodRegistry,
    audit_registry as audit_registry,
    build_default_registry as build_default_registry,
    canonical_simulator_sha256 as canonical_simulator_sha256,
)
from .heuristic_control import (
    BangBangFitter as BangBangFitter,
    HeuristicControlConfig as HeuristicControlConfig,
    RandomReferenceFitter as RandomReferenceFitter,
    SinusoidalFitter as SinusoidalFitter,
)
from .mean_control import (
    MeanControlConfig as MeanControlConfig,
    MeanControlFitter as MeanControlFitter,
)
from .manifest import CanonicalManifest as CanonicalManifest, load_manifest as load_manifest
from .nominal_control import (
    NominalControlConfig as NominalControlConfig,
    NominalControlFitter as NominalControlFitter,
)
from .ppo_control import (
    LegacyPolicyStateError as LegacyPolicyStateError,
    PpoControlConfig as PpoControlConfig,
    PpoControlFitter as PpoControlFitter,
    require_time_only_state_dimension as require_time_only_state_dimension,
)
from .smoke_artifacts import StaleSmokeArtifactError as StaleSmokeArtifactError
from .smoke_runner import (
    SmokeRunReport as SmokeRunReport,
    run_smoke as run_smoke,
)

__all__ = [
    "CanonicalOpenLoopEnv",
    "BangBangFitter",
    "BudgetAmendmentArtifact",
    "BudgetDecision",
    "ControllerFitter",
    "CvarControlConfig",
    "CvarControlFitter",
    "EpisodeParameters",
    "DuplicateMethodRegistrationError",
    "FairnessCertificate",
    "FairnessCheck",
    "FitCost",
    "FitMetadata",
    "FitProvenance",
    "FittedController",
    "HeuristicControlConfig",
    "InvalidProvenanceHashError",
    "LegacyPolicyStateError",
    "MeanControlConfig",
    "MeanControlFitter",
    "MethodRegistration",
    "MethodRegistry",
    "NominalControlConfig",
    "NominalControlFitter",
    "NonTrainingDrawSetError",
    "NonValidationDrawSetError",
    "ObjectiveContribution",
    "PulseController",
    "PulseHorizonMismatchError",
    "PpoControlConfig",
    "PpoControlFitter",
    "PilotStageResult",
    "PlateauCriterion",
    "RandomReferenceFitter",
    "SinusoidalFitter",
    "SmokeRunReport",
    "StaleSmokeArtifactError",
    "TrainingDraws",
    "ValidationDraws",
    "audit_registry",
    "assess_budget",
    "build_default_registry",
    "canonical_simulator_sha256",
    "CanonicalManifest",
    "load_manifest",
    "run_smoke",
    "run_convergence_pilot",
    "require_time_only_state_dimension",
    "select_lower_tail",
]
