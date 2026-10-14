"""KS posterior and separated-state dynamics."""

import torch


def split_state(state, config):
    """
        state: [..., state_dim], concatenated observation Y and posterior p
        output: Y [..., observation_dim], p [..., num_hidden_states]
    """
    m = config.observation_dim
    return state[..., :m], state[..., m:] # split along the last axis; preserve all leading dimensions


def posterior_mean_sensor_and_diffusion(
    state, control, *, problem, config, geometry
    ):
    """
        state: [..., state_dim], control: [..., control_dim]
        output: h_bar [..., observation_dim], B [..., num_hidden_states, observation_dim]
        ... is usually [batch], or [batch, num_candidates] during control optimisation
    """
    y, p = split_state(state, config) # y [..., observation_dim], p [..., num_hidden_states]
    h_values = problem.sensor(control, y) # [..., num_hidden_states, observation_dim], sensor h at each hidden state
    problem.validate_sensor(h_values) # check the hidden-state and observation dimensions

    # p[..., :, None],  shape [..., num_hidden_states, 1              ]
    # h_values,         shape [..., num_hidden_states, observation_dim]
    h_bar = (p[..., :, None] * h_values).sum(dim=-2) # posterior mean of h, [..., observation_dim]

    # p[..., :, None],      shape [..., num_hidden_states, 1              ]
    # h_values,             shape [..., num_hidden_states, observation_dim]
    # h_bar[..., None, :],  shape [..., 1,                 observation_dim]
    B_raw = p[..., :, None] * (h_values - h_bar[..., None, :]) # KS diffusion p_i(h_i - h_bar), [..., num_hidden_states, observation_dim]

    # geometry.projector, shape      [num_hidden_states, num_hidden_states]
    # B_raw,              shape [..., num_hidden_states, observation_dim]
    # project each diffusion column onto the zero-sum tangent space; this preserves mass, not positivity
    B = torch.einsum("ij,...ja->...ia", geometry.projector, B_raw) # [..., num_hidden_states, observation_dim]
    return h_bar, B


def state_dynamics(state, control, *, problem, config, geometry):
    """
        state: [..., state_dim], control: [..., control_dim]
        output: drift [..., state_dim], sigma [..., state_dim, observation_dim], diagnostics
        dS = drift dt + sigma dI, where Y and p share the same innovation Brownian motion I
    """
    _, p = split_state(state, config) # posterior p, [..., num_hidden_states]
    h_bar, B = posterior_mean_sensor_and_diffusion(
        state, control, problem=problem, config=config, geometry=geometry
    )
    drift = torch.cat((h_bar, torch.zeros_like(p)), dim=-1) # [..., state_dim], concatenate observation drift and zero posterior drift

    lead = state.shape[:-1] # leading dimensions, e.g. [batch] or [batch, num_candidates]
    m = config.observation_dim
    eye = torch.eye(m, dtype=state.dtype, device=state.device) # [observation_dim, observation_dim], observation diffusion is identity in this implementation
    eye = eye.reshape(*([1] * len(lead)), m, m).expand(*lead, m, m) # [..., observation_dim, observation_dim], repeat identity over leading dimensions
    sigma = torch.cat((eye, B), dim=-2) # [..., state_dim, observation_dim], stack observation and posterior rows along the state axis

    return drift, sigma, {"B": B, "h_bar": h_bar} # retain the posterior diffusion and observation drift for inspection


def stabilise_posterior(posterior, eps):
    """
        posterior: [..., num_hidden_states]
        output: corrected [..., num_hidden_states], scalar invalid rate
    """
    # flag invalid posterior vectors before correction, using the original numerical tolerances
    invalid = (
        (posterior < -1e-10).any(dim=-1) # at least one entry is below -1e-10
        | # or
        (posterior.sum(dim=-1) - 1.0).abs().gt(1e-8) # total mass differs from one by more than 1e-8
    ) # [...], one flag per posterior vector
    corrected = torch.clamp(posterior, min=eps) # [..., num_hidden_states], set every entry below eps to eps
    corrected = corrected / corrected.sum(dim=-1, keepdim=True) # renormalise each posterior vector to sum to one
    return corrected, invalid.to(posterior.dtype).mean() # scalar fraction of flagged vectors over all leading dimensions, not the fraction of entries changed


def stabilise_state(state, *, config):
    """
        state: [..., state_dim]
        output: corrected state [..., state_dim], scalar invalid rate
    """
    y, p = split_state(state, config)
    p, rate = stabilise_posterior(p, config.posterior_eps) # correct the posterior and record its pre-correction invalid rate
    return torch.cat((y, p), dim=-1), rate # concatenate the unchanged observation with the corrected posterior
