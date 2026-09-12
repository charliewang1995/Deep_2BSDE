"""KS posterior and separated-state dynamics."""

import torch


def split_state(state, config):
    m = config.observation_dim
    return state[..., :m], state[..., m:]


def posterior_mean_sensor_and_diffusion(
    state, control, *, problem, config, geometry
    ):
    y, p = split_state(state, config)
    h_values = problem.sensor(control, y)
    expected = (config.num_hidden_states, config.observation_dim)
    if h_values.shape[-2:] != expected:
        raise ValueError(
            f"sensor must end in {expected}, got {h_values.shape[-2:]}"
        )
    h_bar = (p[..., :, None] * h_values).sum(dim=-2)
    B_raw = p[..., :, None] * (h_values - h_bar[..., None, :])
    B = torch.einsum("ij,...ja->...ia", geometry.projector, B_raw)
    return h_bar, B


def state_dynamics(state, control, *, problem, config, geometry):
    y, p = split_state(state, config)
    h_bar, B = posterior_mean_sensor_and_diffusion(
        state, control, problem=problem, config=config, geometry=geometry
    )
    drift = torch.cat((h_bar, torch.zeros_like(p)), dim=-1)

    lead = state.shape[:-1]
    m = config.observation_dim
    eye = torch.eye(m, dtype=state.dtype, device=state.device)
    eye = eye.reshape(*([1] * len(lead)), m, m).expand(*lead, m, m)
    sigma = torch.cat((eye, B), dim=-2)
    return drift, sigma, {"B": B, "h_bar": h_bar}


def stabilise_posterior(posterior, eps):
    # this is a simple stabilisation that clips negative entries and renormalises
    # complicated stabilisation could be used
    invalid = (
        (posterior < -1e-10).any(dim=-1) # negetive entries
        | # or 
        (posterior.sum(dim=-1) - 1.0).abs().gt(1e-8) # sum not equal to 1
    )
    corrected = torch.clamp(posterior, min=eps)
    corrected = corrected / corrected.sum(dim=-1, keepdim=True)
    return corrected, invalid.to(posterior.dtype).mean()


def stabilise_state(state, *, config):
    y, p = split_state(state, config)
    p, rate = stabilise_posterior(p, config.posterior_eps)
    return torch.cat((y, p), dim=-1), rate
