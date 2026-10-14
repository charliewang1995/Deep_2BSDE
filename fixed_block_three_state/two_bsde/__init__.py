"""Public API for the reusable geometric 2BSDE solver."""
from .config import ModelConfig, HamiltonianConfig, TrainingConfig
from .problem import BayesianControlProblem
from .geometry import SimplexGeometry, make_simplex_tangent_basis
from .dynamics import (
    split_state,
    posterior_mean_sensor_and_diffusion,
    state_dynamics,
    stabilise_posterior,
    stabilise_state,
)
from .networks import GeometricParallelBlock, GeometricTimeConditionedBlock
from .hamiltonian import HamiltonianOptimizer
from .model import (
    GeometricClassicalFixedAux2BSDE,
    GeometricSharedTimeFixedAux2BSDE,
)
from .training import TrainingHistory, train
from .evaluation import EvaluationResult, evaluate

__all__ = [
    "BayesianControlProblem",
    "EvaluationResult",
    "GeometricClassicalFixedAux2BSDE",
    "GeometricParallelBlock",
    "GeometricSharedTimeFixedAux2BSDE",
    "GeometricTimeConditionedBlock",
    "HamiltonianConfig",
    "HamiltonianOptimizer",
    "ModelConfig",
    "SimplexGeometry",
    "TrainingConfig",
    "TrainingHistory",
    "evaluate",
    "make_simplex_tangent_basis",
    "posterior_mean_sensor_and_diffusion",
    "split_state",
    "stabilise_posterior",
    "stabilise_state",
    "state_dynamics",
    "train",
]
