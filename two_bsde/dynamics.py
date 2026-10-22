"""KS dynamics for constant isotropic observation noise sigma > 0."""

import torch


def split_state(state, config):
    """
        state: [..., state_dim], concatenated observation Y and posterior p
        output: Y [..., observation_dim], p [..., num_hidden_states]
    """
    m = config.observation_dim
    return state[..., :m], state[..., m:] # split along the last axis; preserve all leading dimensions


def _broadcast_state_and_control(state, control, config):
    """Broadcast only batch axes; retain separate state and control axes."""
    if state.ndim < 1 or state.shape[-1] != config.state_dim:
        raise ValueError(
            f"state must end in ({config.state_dim},), got {tuple(state.shape)}"
        )
    if control.ndim < 1 or control.shape[-1] != config.control_dim:
        raise ValueError(
            f"control must end in ({config.control_dim},), got {tuple(control.shape)}"
        )
    try:
        batch_shape = torch.broadcast_shapes(state.shape[:-1], control.shape[:-1])
    except RuntimeError as exc:
        raise ValueError(
            "state and control batch axes must be broadcastable; got "
            f"{tuple(state.shape[:-1])} and {tuple(control.shape[:-1])}. "
            "For a candidate axis, use state[:, None, :] and controls [batch, K, control_dim]."
        ) from exc
    return (
        state.expand(*batch_shape, config.state_dim),
        control.expand(*batch_shape, config.control_dim),
    )


def posterior_mean_sensor_and_diffusion(
    state, control, *, problem, config, geometry
    ):
    """
        state: [..., state_dim], control: [..., control_dim]
        output: h_bar [..., observation_dim], B [..., num_hidden_states, observation_dim]
        ... is usually [batch], or [batch, num_candidates] during control optimisation.
        A shared control [control_dim] is expanded over the state batch axes.
        For per-path controls, pass [batch, control_dim]. For candidate controls,
        pass state [batch, 1, state_dim] and control [batch, K, control_dim],
        or explicitly expand both inputs to the same batch shape.
    """
    state, control = _broadcast_state_and_control(state, control, config)
    y, p = split_state(state, config) # y [..., observation_dim], p [..., num_hidden_states]
    h_values = problem.sensor(control, y) # [..., num_hidden_states, observation_dim], sensor h at each hidden state
    problem.validate_sensor(h_values, batch_shape=state.shape[:-1])
    h_values = h_values.expand(
        *state.shape[:-1], config.num_hidden_states, config.observation_dim
    )

    # p[..., :, None],  shape [..., num_hidden_states, 1              ]
    # h_values,         shape [..., num_hidden_states, observation_dim]
    h_bar = (p[..., :, None] * h_values).sum(dim=-2) # posterior mean of h, [..., observation_dim]

    # p[..., :, None],      shape [..., num_hidden_states, 1              ]
    # h_values,             shape [..., num_hidden_states, observation_dim]
    # h_bar[..., None, :],  shape [..., 1,                 observation_dim]
    # Standard innovation dI = (dY - h_bar dt) / observation_noise_std.
    # Hence dp_i = p_i (h_i - h_bar) / observation_noise_std dI.
    B_raw = p[..., :, None] * (h_values - h_bar[..., None, :]) / config.observation_noise_std

    # geometry.projector, shape      [num_hidden_states, num_hidden_states]
    # B_raw,              shape [..., num_hidden_states, observation_dim]
    # project each diffusion column onto the zero-sum tangent space; this preserves mass, not positivity
    B = torch.einsum("ij,...ja->...ia", geometry.projector, B_raw) # [..., num_hidden_states, observation_dim]
    return h_bar, B


def state_dynamics(state, control, *, problem, config, geometry):
    """
        state: [..., state_dim], control: [..., control_dim]
        output: drift [..., state_dim], sigma [..., state_dim, observation_dim], diagnostics
        dS = drift dt + sigma dI, where Y and p share standard innovation Brownian motion I.
        The observation block is observation_noise_std * identity;
        the posterior block is p_i(h_i - h_bar) / observation_noise_std.
    """
    h_bar, B = posterior_mean_sensor_and_diffusion(
        state, control, problem=problem, config=config, geometry=geometry
    )
    drift = torch.cat((h_bar, torch.zeros_like(B[..., 0])), dim=-1) # [..., state_dim], observation drift and zero posterior drift with the broadcast batch shape

    lead = h_bar.shape[:-1] # broadcast leading dimensions, e.g. [batch] or [batch, num_candidates]
    m = config.observation_dim
    eye = config.observation_noise_std * torch.eye(m, dtype=state.dtype, device=state.device)
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
