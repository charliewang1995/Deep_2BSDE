"""Application-specific problem interface."""
from abc import ABC, abstractmethod
import torch


class BayesianControlProblem(ABC):
    x_support: torch.Tensor # [num_hidden_states, hidden_dim]
    initial_observation: torch.Tensor # [observation_dim]
    initial_posterior: torch.Tensor # [num_hidden_states]
    control_low: torch.Tensor # [control_dim], lowest value in each dimension
    control_high: torch.Tensor # [control_dim], largest value in each dimension 
    auxiliary_control_path: torch.Tensor # [num_time_steps, control_dim]

    def __init__(self, config):
        self.config = config

    def validate(self, config): # check consistency of the problem with config.py
        if self.x_support.shape[0] != (
            config.num_hidden_states
        ):
            raise ValueError(
                "x_support must contain "
                f"{config.num_hidden_states} hidden states, "
                f"got {self.x_support.shape[0]}"
            )

        if self.x_support.shape[1] != (
                    config.hidden_dim
                ):
                    raise ValueError(
                        "x_support must have dimension "
                        f"{config.hidden_dim}, "
                        f"got {self.x_support.shape[1]}"
                    )

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
        
    
    @abstractmethod
    def sensor(self, control, observation):
        '''
            The sensor h evaluated at each hidden state, and regarded as a single tensor. 
            output [batch, num_hidden_states, observation_dim]
        '''
        pass

    def validate_sensor(self, h_values):
        expected = (self.config.num_hidden_states, self.config.observation_dim)
        if h_values.shape[-2:] != expected:
            raise ValueError(
                f"sensor must end in {expected}, got {h_values.shape[-2:]}"
            ) 

    @abstractmethod
    def running_cost(self, state, control):
        """
            output: [batch, num_time_steps]
        """
        pass

    @abstractmethod
    def terminal_cost(self, state): 
        """
            output: [batch, ]
        """
        pass

    def terminal_gradient(self, state):
        with torch.enable_grad():
            s = state.detach().clone().requires_grad_(True)
            v = self.terminal_cost(s)
            grad = torch.autograd.grad(v.sum(), s)[0]
        return grad.detach()
