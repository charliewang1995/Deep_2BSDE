"""Classical fixed-auxiliary geometric 2BSDE model."""
import torch
from torch import nn
from .dynamics import state_dynamics, stabilise_state
from .networks import GeometricParallelBlock


class GeometricClassicalFixedAux2BSDE(nn.Module):
    def __init__(
        self, *, problem, config, geometry, hamiltonian_optimizer
    ):
        super().__init__()
        self.problem = problem
        self.config = config
        self.geometry = geometry
        self.hamiltonian_optimizer = hamiltonian_optimizer

        self.blocks = nn.ModuleList(
            GeometricParallelBlock(config, geometry)
            for _ in range(max(config.num_time_steps - 1, 0))
        )

        self.initial_value = nn.Parameter(
            torch.tensor(float(config.value_initial_guess))
        )
        self.initial_gradient_raw = nn.Parameter(
            torch.zeros(config.reduced_vector_dim)
        )
        self.initial_drift_gradient_raw = nn.Parameter(
            torch.zeros(config.reduced_vector_dim) #A0
        )
        self.initial_gamma_raw = nn.Parameter(
            config.gamma_initial_scale
            * torch.randn(config.gamma_parameter_dim)
        )

    def initial_gradient(self):
        return self.geometry.vector_from_raw(self.initial_gradient_raw)

    def initial_drift_gradient(self):
        return self.geometry.vector_from_raw(self.initial_drift_gradient_raw)

    def initial_gamma(self):
        return self.geometry.gamma_from_raw(self.initial_gamma_raw)

    def forward(self, innovations, return_paths=False):
        cfg = self.config
        if innovations.ndim != 3:
            raise ValueError("innovations must be [batch,time,observation_dim]")
        if innovations.shape[1:] != (
            cfg.num_time_steps, cfg.observation_dim
        ):
            raise ValueError("innovation shape mismatch")

        batch = innovations.shape[0]
        dtype, device = innovations.dtype, innovations.device

        y0 = self.problem.initial_observation.to(dtype=dtype, device=device)
        p0 = self.problem.initial_posterior.to(dtype=dtype, device=device)
        S0 = torch.cat((y0, p0))
        state = S0[None, :].expand(batch, -1).clone()

        R = self.initial_value.to(dtype=dtype, device=device).expand(batch)
        Z = self.initial_gradient()[None, :].expand(batch, -1)
        A = self.initial_drift_gradient()[None, :].expand(batch, -1)
        Gamma = self.initial_gamma()[None, :, :].expand(batch, -1, -1)

        S_path = [state]
        R_path = [R]
        Z_path = [Z]
        A_path, G_path, u_aux_path, u_star_path = [], [], [], []
        correction_rates = []

        aux_path = self.problem.auxiliary_control_path.to(dtype=dtype, device=device)

        for step in range(cfg.num_time_steps):
            u_aux = aux_path[step][None, :].expand(batch, -1)

            drift_aux, sigma_aux, _ = state_dynamics(
                state, u_aux,
                problem=self.problem, config=cfg, geometry=self.geometry
            )

            h_min, u_star = self.hamiltonian_optimizer.minimise(
                state, 
                Z, 
                Gamma
            )

            aux_quad = torch.einsum(
                "bia,bij,bja->b", sigma_aux, Gamma, sigma_aux
            )
            driver = (
                -h_min
                + (drift_aux * Z).sum(dim=-1)
                + 0.5 * aux_quad
            )
            Z_sigma = torch.einsum("bi,bia->ba", Z, sigma_aux)

            dI = innovations[:, step, :]
            R = R + driver * cfg.dt + (Z_sigma * dI).sum(dim=-1)

            Z = (
                Z
                + A * cfg.dt
                + torch.einsum(
                    "bij,bja,ba->bi", Gamma, sigma_aux, dI
                )
            )
            Z = self.geometry.project_gradient(Z)

            state = (
                state
                + drift_aux * cfg.dt
                + torch.einsum("bia,ba->bi", sigma_aux, dI)
            )
            state, corr = stabilise_state(state, config=cfg)

            A_path.append(A)
            G_path.append(Gamma)
            u_aux_path.append(u_aux)
            u_star_path.append(u_star)
            correction_rates.append(corr)
            S_path.append(state)
            R_path.append(R)
            Z_path.append(Z)

            if step < cfg.num_time_steps - 1:
                A, Gamma = self.blocks[step](state)

        correction_rate = torch.stack(correction_rates).mean()

        if not return_paths:
            return R, Z, state, correction_rate

        return {
            "R_T": R,
            "Z_T": Z,
            "S_T": state,
            "R_path": torch.stack(R_path, dim=1),
            "Z_path": torch.stack(Z_path, dim=1),
            "S_path": torch.stack(S_path, dim=1),
            "A_path": torch.stack(A_path, dim=1),
            "Gamma_path": torch.stack(G_path, dim=1),
            "u_aux_path": torch.stack(u_aux_path, dim=1),
            "u_star_path": torch.stack(u_star_path, dim=1),
            "posterior_correction_rate": correction_rate,
        }

    def save_blocks(self, path):
        torch.save(self.blocks.state_dict(), path)

    def load_blocks(self, path, *, map_location="cpu"):
        self.blocks.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True)
        )
