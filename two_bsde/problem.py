"""Application-specific problem interface."""
from abc import ABC, abstractmethod
import torch


class BayesianControlProblem(ABC):
    x_support: torch.Tensor
    initial_observation: torch.Tensor
    initial_posterior: torch.Tensor
    control_low: torch.Tensor
    control_high: torch.Tensor
    auxiliary_control_path: torch.Tensor

    def __init__(
        self,
        config,
    ):
        self.config = config

    def validate(self, config):

        if self.initial_observation.shape != (
            config.observation_dim,
        ):
            raise ValueError(
                "initial_observation must have shape "
                f"({config.observation_dim},), "
                f"got {self.initial_observation.shape}"
            )

        if self.initial_posterior.shape != (
            config.num_hidden_states,
        ):
            raise ValueError(
                "initial_posterior must have shape "
                f"({config.num_hidden_states},), "
                f"got {self.initial_posterior.shape}"
            )

        if self.control_low.shape != (
            config.control_dim,
        ):
            raise ValueError(
                "control_low must have shape "
                f"({config.control_dim},), "
                f"got {self.control_low.shape}"
            )

        if self.control_high.shape != (
            config.control_dim,
        ):
            raise ValueError(
                "control_high must have shape "
                f"({config.control_dim},), "
                f"got {self.control_high.shape}"
            )

        if self.auxiliary_control_path.shape != (
            config.num_time_steps,
            config.control_dim,
        ):
            raise ValueError(
                "auxiliary_control_path must have shape "
                f"({config.num_time_steps}, "
                f"{config.control_dim}), "
                f"got {self.auxiliary_control_path.shape}"
            )

        if self.x_support.shape[0] != (
            config.num_hidden_states
        ):
            raise ValueError(
                "x_support must contain "
                f"{config.num_hidden_states} hidden states, "
                f"got {self.x_support.shape[0]}"
        )
    
    @abstractmethod
    def sensor(self, control, observation):
        pass

    @abstractmethod
    def running_cost(self, state, control):
        pass

    @abstractmethod
    def terminal_cost(self, state):
        pass

    def terminal_gradient(self, state):
        with torch.enable_grad():
            s = state.detach().clone().requires_grad_(True)
            v = self.terminal_cost(s)
            grad = torch.autograd.grad(v.sum(), s)[0]
        return grad.detach()
