"""Reusable neural blocks for A_k and Gamma_k."""
import torch
from torch import nn


def make_mlp(input_dim, output_dim, width, depth):
    """
        Returned network input: [batch, input_dim]
        Returned network output: [batch, output_dim]
    """
    layers = []
    current = input_dim
    for _ in range(depth): # depth hidden layers, each with width neurons and Tanh activation
        layers.extend((nn.Linear(current, width), nn.Tanh()))
        current = width
    layers.append(nn.Linear(current, output_dim)) # linear output layer, no activation
    return nn.Sequential(*layers)


class GeometricParallelBlock(nn.Module): # two separate networks taking the same state S = (Y, p)
    def __init__(self, config, geometry):
        super().__init__()
        self.geometry = geometry
        self.a_net = make_mlp(
            config.state_dim, 
            config.reduced_vector_dim, # observation coordinates and posterior tangent coordinates of A
            config.network_width, 
            config.network_depth
        )
        self.gamma_net = make_mlp(
            config.state_dim, 
            config.gamma_parameter_dim, # independent coefficients of the reduced symmetric Gamma
            config.network_width, 
            config.network_depth
        )

    def forward(self, state):
        """
            state: [batch, state_dim], current observation and posterior
            output: A [batch, state_dim], Gamma [batch, state_dim, state_dim]
        """
        # a_net(state),     shape [batch, reduced_vector_dim]
        # gamma_net(state), shape [batch, gamma_parameter_dim]
        A = self.geometry.vector_from_raw(self.a_net(state)) # drift of Z, posterior part is zero-sum
        Gamma = self.geometry.gamma_from_raw(self.gamma_net(state)) # Hessian approximation, symmetric with tangent posterior blocks
        return A, Gamma

class TimeGeometricBlock(nn.Module):
    """
        Shared networks for V(t, S), Gamma(t, S) and A(t, S, u_aux).
        Time is normalised by the horizon and included before the state coordinates.
        A also receives the auxiliary control because the drift of Z depends on it.
    """

    def __init__(self, config, geometry):
        super().__init__()
        if config.horizon <= 0:
            raise ValueError("horizon must be positive")
        if config.gamma_initial_scale < 0:
            raise ValueError("gamma_initial_scale must be nonnegative")
        self.config = config
        self.geometry = geometry
        input_dim = config.state_dim + 1 # normalised time and state S = (Y, p)
        self.value_net = make_mlp(
            input_dim, 1, config.network_width, config.network_depth
        )
        self.gamma_net = make_mlp(
            input_dim, config.gamma_parameter_dim,
            config.network_width, config.network_depth
        )
        self.a_net = make_mlp(
            input_dim + config.control_dim, config.reduced_vector_dim,
            config.network_width, config.network_depth
        )

        # Small nonzero value weights allow initial dependence on time and state.
        nn.init.normal_(self.value_net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.value_net[-1].bias, config.value_initial_guess)
        gamma_head = self.gamma_net[-1]
        nn.init.normal_(
            gamma_head.weight, mean=0.0,
            std=config.gamma_initial_scale / gamma_head.in_features ** 0.5
        ) # scale by the number of inputs to the final layer
        nn.init.zeros_(gamma_head.bias)

    def _time_state(self, time, state):
        """
            state: [..., state_dim], including a single state [state_dim]
            time: scalar or tensor with shape state.shape[:-1]
            output: [..., state_dim+1]
        """
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a tensor")
        if state.ndim < 1 or state.shape[-1] != self.config.state_dim:
            raise ValueError("state must have shape [..., state_dim]")
        if not torch.is_floating_point(state):
            raise TypeError("state must have a floating-point dtype")
        t = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        if t.ndim == 0:
            t = t.expand(state.shape[:-1]) # use the same time for all states
        elif t.shape != state.shape[:-1]:
            raise ValueError("time must be scalar or have shape state.shape[:-1]")
        return torch.cat((t.unsqueeze(-1) / self.config.horizon, state), dim=-1)

    def value(self, time, state):
        """
            state: [..., state_dim]
            output: [...], direct value approximation at (time, state)
        """
        return self.value_net(self._time_state(time, state)).squeeze(-1)

    def gamma(self, time, state):
        """
            state: [..., state_dim]
            output: [..., state_dim, state_dim], symmetric with tangent posterior blocks
        """
        raw = self.gamma_net(self._time_state(time, state)) # [..., gamma_parameter_dim]
        return self.geometry.gamma_from_raw(raw)

    def drift_gradient(self, time, state, control):
        """
            state: [..., state_dim]
            control: [control_dim] or [..., control_dim], the auxiliary control
            output: [..., state_dim], drift A of Z with zero-sum posterior part
        """
        features = self._time_state(time, state)
        control = torch.as_tensor(control, dtype=state.dtype, device=state.device)
        expected = state.shape[:-1] + (self.config.control_dim,)
        if control.shape == (self.config.control_dim,):
            control = control.expand(expected) # broadcast one control over the states
        elif control.shape != expected:
            raise ValueError("control must have shape [control_dim] or [..., control_dim] matching state")
        raw = self.a_net(torch.cat((features, control), dim=-1)) # [..., reduced_vector_dim]
        return self.geometry.vector_from_raw(raw)

    def forward(self, time, state, control):
        """
            output: V [...], A [..., state_dim], Gamma [..., state_dim, state_dim]
        """
        return self.value(time, state), self.drift_gradient(time, state, control), self.gamma(time, state)
