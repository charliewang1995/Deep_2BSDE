"""Configuration objects for the reusable geometric 2BSDE solver."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    horizon: float = 1.0
    hidden_dim: int = 1
    observation_dim: int = 1
    control_dim: int = 1
    num_hidden_states: int = 3
    num_time_steps: int = 12
    network_width: int = 32
    network_depth: int = 2
    value_initial_guess: float = 0.25
    gamma_initial_scale: float = 0.02
    posterior_eps: float = 1e-8

    @property
    def dt(self):
        return self.horizon / self.num_time_steps

    @property
    def state_dim(self):
        return self.observation_dim + self.num_hidden_states

    @property
    def tangent_dim(self):
        return self.num_hidden_states - 1

    @property
    def reduced_vector_dim(self):
        return self.observation_dim + self.tangent_dim

    @property
    def gamma_parameter_dim(self):
        m, q = self.observation_dim, self.tangent_dim
        return m * (m + 1) // 2 + m * q + q * (q + 1) // 2


@dataclass(frozen=True)
class HamiltonianConfig:
    num_starts: int = 4
    num_steps: int = 10
    learning_rate: float = 8e-2
    gradient_clip: float = 20.0
    seed: int = 271828


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 256
    initial_value_lr: float = 1e-4
    learning_rate: float = 1e-3
    gradient_clip: float = 10.0
    value_loss_weight: float = 1.0
    gradient_loss_weight: float = 1.0
