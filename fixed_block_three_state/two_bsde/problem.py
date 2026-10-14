"""Application-specific problem interface."""
from abc import ABC, abstractmethod
import torch


class BayesianControlProblem(ABC):
    """Interface consumed by the generic KS/2BSDE solver."""

    def __init__(self, config):
        self.config = config

    x_support: torch.Tensor
    initial_observation: torch.Tensor
    initial_posterior: torch.Tensor
    control_low: torch.Tensor
    control_high: torch.Tensor
    auxiliary_control_path: torch.Tensor

    def validate(self, config=None):
        cfg = self.config if config is None else config

        expected = {
            "initial_observation": (cfg.observation_dim,),
            "initial_posterior": (cfg.num_hidden_states,),
            "control_low": (cfg.control_dim,),
            "control_high": (cfg.control_dim,),
            "auxiliary_control_path": (cfg.num_time_steps, cfg.control_dim),
            "x_support": (cfg.num_hidden_states, cfg.hidden_dim),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")

        if torch.any(self.initial_posterior < 0):
            raise ValueError("initial_posterior must be non-negative")
        one = torch.ones(
            (), dtype=self.initial_posterior.dtype,
            device=self.initial_posterior.device,
        )
        if not torch.isclose(self.initial_posterior.sum(), one, atol=1e-8):
            raise ValueError("initial_posterior must sum to one")
        if torch.any(self.control_low > self.control_high):
            raise ValueError("control_low must be <= control_high")
        if torch.any(self.auxiliary_control_path < self.control_low) or torch.any(
            self.auxiliary_control_path > self.control_high
        ):
            raise ValueError("auxiliary_control_path must satisfy the control bounds")

    @abstractmethod
    def sensor(self, control, observation):
        """Return h(u,y,x_i), ending in dimensions (num_hidden_states, observation_dim)."""
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
