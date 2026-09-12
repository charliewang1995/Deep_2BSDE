"""Reusable neural blocks for A_k and Gamma_k."""
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
    def __init__(self, config, geometry):
        super().__init__()
        self.geometry = geometry
        self.a_net = make_mlp(
            config.state_dim, 
            config.reduced_vector_dim,
            config.network_width, 
            config.network_depth
        )
        self.gamma_net = make_mlp(
            config.state_dim, 
            config.gamma_parameter_dim,
            config.network_width, 
            config.network_depth
        )

    def forward(self, state):
        A = self.geometry.vector_from_raw(self.a_net(state))
        Gamma = self.geometry.gamma_from_raw(self.gamma_net(state))
        return A, Gamma
