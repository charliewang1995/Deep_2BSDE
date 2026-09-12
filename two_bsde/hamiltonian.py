"""Hamiltonian evaluation and inner control minimisation."""
import torch
from torch import nn, optim
from .dynamics import state_dynamics


class HamiltonianOptimizer:
    def __init__(
        self, *, problem, model_config, optimizer_config, geometry
    ):
        self.problem = problem
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.geometry = geometry

    def evaluate(self, state, gradient, gamma, control):
        drift, sigma, _ = state_dynamics(
            state, control,
            problem=self.problem,
            config=self.model_config,
            geometry=self.geometry,
        )
        running = self.problem.running_cost(state, control)
        linear = (drift * gradient).sum(dim=-1)
        quadratic = 0.5 * torch.einsum(
            "...ia,...ij,...ja->...",
            sigma, gamma, sigma
        )
        return running + linear + quadratic

    def project_control(self, control):
        lo = self.problem.control_low.to(dtype=control.dtype, device=control.device)
        hi = self.problem.control_high.to(dtype=control.dtype, device=control.device)
        return torch.maximum(torch.minimum(control, hi), lo)

    def _sobol_starts(self, num_starts, *, dtype, device):
        n = self.model_config.control_dim
        engine = torch.quasirandom.SobolEngine(
            dimension=n, scramble=True, seed=self.optimizer_config.seed
        )
        unit = engine.draw(num_starts).to(dtype=dtype, device=device)
        lo = self.problem.control_low.to(dtype=dtype, device=device)
        hi = self.problem.control_high.to(dtype=dtype, device=device)
        starts = lo + (hi - lo) * unit
        starts[0] = 0.5 * (lo + hi)
        return self.project_control(starts)

    def minimise(self, state, gradient, gamma):
        if state.ndim != 2:
            raise ValueError("minimise expects [batch,state_dim]")

        # For a scalar box control, a dense vectorised grid is both faster
        # and more reproducible than launching an inner optimiser at every
        # time step. The final H is still re-evaluated at the selected
        # control with the original gradient/Gamma tensors.
        if self.model_config.control_dim == 1:
            batch = state.shape[0]
            lo = float(self.problem.control_low.item())
            hi = float(self.problem.control_high.item())
            K = max(81, 20 * self.optimizer_config.num_starts + 1)
            grid = torch.linspace(
                lo, hi, K, dtype=state.dtype, device=state.device
            )
            controls = grid[None, :, None].expand(batch, -1, -1)
            s = state.detach()[:, None, :].expand(-1, K, -1)
            z = gradient.detach()[:, None, :].expand(-1, K, -1)
            g = gamma.detach()[:, None, :, :].expand(-1, K, -1, -1)
            with torch.no_grad():
                values = self.evaluate(s, z, g, controls)
                idx = values.argmin(dim=1)
                bidx = torch.arange(batch, device=state.device)
                u_star = controls[bidx, idx].detach()
            h_min = self.evaluate(state, gradient, gamma, u_star)
            return h_min, u_star

        K = self.optimizer_config.num_starts
        batch = state.shape[0]
        starts = self._sobol_starts(
            K, dtype=state.dtype, device=state.device
        )[None, :, :].expand(batch, -1, -1).clone()
        variable = nn.Parameter(starts)
        inner = optim.Adam([variable], lr=self.optimizer_config.learning_rate)
        s = state.detach()[:, None, :].expand(-1, K, -1)
        z = gradient.detach()[:, None, :].expand(-1, K, -1)
        g = gamma.detach()[:, None, :, :].expand(-1, K, -1, -1)
        with torch.enable_grad():
            for _ in range(self.optimizer_config.num_steps):
                u = self.project_control(variable)
                values = self.evaluate(s, z, g, u)
                loss = values.mean()
                inner.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [variable], self.optimizer_config.gradient_clip
                )
                inner.step()
                with torch.no_grad():
                    variable.copy_(self.project_control(variable))
        with torch.no_grad():
            candidates = self.project_control(variable)
            values = self.evaluate(s, z, g, candidates)
            idx = values.argmin(dim=1)
            bidx = torch.arange(batch, device=state.device)
            u_star = candidates[bidx, idx].detach()
        h_min = self.evaluate(state, gradient, gamma, u_star)
        return h_min, u_star
