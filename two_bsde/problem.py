"""Application-specific problem interface."""
from abc import ABC, abstractmethod
import torch


class BayesianControlProblem(ABC):
    x_support: torch.Tensor # [num_hidden_states, hidden_dim]
    initial_observation: torch.Tensor # [observation_dim]
    initial_posterior: torch.Tensor # [num_hidden_states]
    control_low: torch.Tensor # [control_dim], lowest value in each dimension
    control_high: torch.Tensor # [control_dim], largest value in each dimension 
    auxiliary_control_path: torch.Tensor # optional [num_time_steps, control_dim], used only by the fixed-auxiliary model

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

        auxiliary_control_path = getattr(self, "auxiliary_control_path", None)
        if auxiliary_control_path is not None and auxiliary_control_path.shape != (
            config.num_time_steps,
            config.control_dim,
        ):
            raise ValueError(
                "auxiliary_control_path must have shape "
                f"({config.num_time_steps}, "
                f"{config.control_dim}), "
                f"got {auxiliary_control_path.shape}"
            )
        
    
    @abstractmethod
    def sensor(self, control, observation):
        """Evaluate the sensor at every hidden state without mixing paths.

        Inputs: control [..., control_dim], observation [..., observation_dim].
        Output: [..., num_hidden_states, observation_dim]. Dynamics broadcasts
        the input batch axes before calling this method: ... is [batch] during
        a rollout and [batch, num_candidates] during Hamiltonian optimisation.
        Preserve these leading axes, including axes of length one. Explicit
        singleton batch axes in the output may broadcast to the input axes.

        For scalar observation/control and h(u, x_i) = u * x_i, use:

            x = self.x_support[:, 0].to(control)
            return control[..., 0, None, None] * x[:, None]

        This returns [..., num_hidden_states, 1]. Do not use control[0],
        control.item(), or a mean over paths: each path has its own control.
        """
        pass

    def validate_sensor(self, h_values, *, batch_shape=None):
        expected = (self.config.num_hidden_states, self.config.observation_dim)
        if h_values.ndim < 2 or h_values.shape[-2:] != expected:
            raise ValueError(
                f"sensor must end in {expected}, got {h_values.shape[-2:]}"
            ) 
        if batch_shape is not None:
            batch_shape = tuple(batch_shape)
            actual_batch = tuple(h_values.shape[:-2])
            if len(actual_batch) != len(batch_shape) or any(
                actual not in (1, requested)
                for actual, requested in zip(actual_batch, batch_shape)
            ):
                raise ValueError(
                    "sensor must preserve every input batch axis: expected "
                    f"{batch_shape + expected} (singleton batch axes may broadcast), "
                    f"got {tuple(h_values.shape)}. Use control[..., ...] to retain "
                    "path and candidate axes."
                )

    @abstractmethod
    def running_cost(self, state, control):
        """
            state [..., state_dim], control [..., control_dim]; output [...].
            Preserve the path and any candidate-control axes independently.
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
