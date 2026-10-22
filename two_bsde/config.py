# global parameter setting

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    horizon: float = 1.0 # time horizion
    hidden_dim: int = 1 # dimension of hidden state X
    observation_dim: int = 1 # dimension of observation Y
    control_dim: int = 1    # dimension of control u
    num_hidden_states: int = 3 # number of hidden states, i.e. number of states in the finite state problem
    num_time_steps: int = 12 
    network_width: int = 64
    network_depth: int = 3
    value_initial_guess: float = 0.01 # initial guess for the value function, also the initial value for V0 
    gamma_initial_scale: float = 0.02 # 
    posterior_eps: float = 1e-8 # correcting bound for posterior dynamic
    observation_noise_std: float = 1.0 # positive scalar sigma in dY = h dt + sigma dW

    def __post_init__(self):
        if not math.isfinite(self.observation_noise_std) or self.observation_noise_std <= 0:
            raise ValueError("observation_noise_std must be finite and strictly positive")

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
class HamiltonianConfig: # hamiltonian optimiser setup
    num_starts: int = 12
    num_steps: int = 30
    learning_rate: float = 5e-2
    gradient_clip: float = 20.0
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig: # training setup
    epochs: int = 500
    batch_size: int = 128
    initial_value_lr: float = 1e-4
    learning_rate: float = 1e-3
    gradient_clip: float = 10.0
    value_loss_weight: float = 1.0
    gradient_loss_weight: float = 1.0
