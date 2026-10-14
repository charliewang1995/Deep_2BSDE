"""Reusable geometric neural blocks for A and Gamma."""
import torch
from torch import nn


def make_mlp(input_dim, output_dim, width, depth):
    layers = []
    current = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, width), nn.Tanh()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class GeometricParallelBlock(nn.Module):
    """Classical time-slice block: state -> (A_k, Gamma_k)."""

    def __init__(self, config, geometry):
        super().__init__()
        self.geometry = geometry
        self.a_net = make_mlp(
            config.state_dim, config.reduced_vector_dim,
            config.network_width, config.network_depth
        )
        self.gamma_net = make_mlp(
            config.state_dim, config.gamma_parameter_dim,
            config.network_width, config.network_depth
        )

    def forward(self, state):
        A = self.geometry.vector_from_raw(self.a_net(state))
        Gamma = self.geometry.gamma_from_raw(self.gamma_net(state))
        return A, Gamma


class GeometricTimeConditionedBlock(nn.Module):
    """Shared block representing one Markovian map (t,state) -> (A,Gamma).

    Time is normalised by the horizon before being concatenated with the
    ambient state.  The outputs are still represented intrinsically and
    reconstructed by ``SimplexGeometry``.
    """

    def __init__(self, config, geometry):
        super().__init__()
        self.config = config
        self.geometry = geometry
        input_dim = 1 + config.state_dim
        self.a_net = make_mlp(
            input_dim, config.reduced_vector_dim,
            config.network_width, config.network_depth
        )
        self.gamma_net = make_mlp(
            input_dim, config.gamma_parameter_dim,
            config.network_width, config.network_depth
        )

    def _time_feature(self, time, state):
        tau = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        tau = tau / self.config.horizon
        lead = state.shape[:-1]
        if tau.ndim == 0:
            tau = tau.expand(*lead)
        else:
            tau = torch.broadcast_to(tau, lead)
        return tau.unsqueeze(-1)

    def forward(self, time, state):
        features = torch.cat((self._time_feature(time, state), state), dim=-1)
        A = self.geometry.vector_from_raw(self.a_net(features))
        Gamma = self.geometry.gamma_from_raw(self.gamma_net(features))
        return A, Gamma
