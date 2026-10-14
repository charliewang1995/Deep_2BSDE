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
        """
            state, gradient: [batch, state_dim]
            gamma: [batch, state_dim, state_dim], control: [batch, control_dim]
            output: [batch], the Hamiltonian at the given control
            An extra K axis can be included after batch to evaluate K controls.
        """
        drift, sigma, _ = state_dynamics(
            state, control,
            problem=self.problem,
            config=self.model_config,
            geometry=self.geometry,
        ) # drift [batch, state_dim], sigma [batch, state_dim, observation_dim]
        running = self.problem.running_cost(state, control) # [batch]
        linear = (drift * gradient).sum(dim=-1) # [batch], drift dotted with the value gradient

        # sigma, shape [batch, state_dim, observation_dim]
        # gamma, shape [batch, state_dim, state_dim]
        # i, j index state coordinates, a indexes Brownian coordinates
        quadratic = 0.5 * torch.einsum(
            "...ia,...ij,...ja->...",
            sigma, gamma, sigma
        ) # [batch], 0.5 * trace(sigma.T @ gamma @ sigma)
        return running + linear + quadratic

    def project_control(self, control): # clip each control component to its admissible interval
        lo = self.problem.control_low.to(dtype=control.dtype, device=control.device) # [control_dim]
        hi = self.problem.control_high.to(dtype=control.dtype, device=control.device) # [control_dim]
        return torch.maximum(torch.minimum(control, hi), lo)

    def _sobol_starts(self, num_starts, *, dtype, device):
        n = self.model_config.control_dim
        engine = torch.quasirandom.SobolEngine(
            dimension=n, scramble=True, seed=self.optimizer_config.seed
        ) # fixed seed gives the same scrambled Sobol starting points on each call
        unit = engine.draw(num_starts).to(dtype=dtype, device=device) # [num_starts, control_dim], points in the unit box
        lo = self.problem.control_low.to(dtype=dtype, device=device)
        hi = self.problem.control_high.to(dtype=dtype, device=device)
        starts = lo + (hi - lo) * unit # [num_starts, control_dim], scale to the control box
        starts[0] = 0.5 * (lo + hi) # replace the first start by the midpoint
        return self.project_control(starts)

    def minimise(self, state, gradient, gamma):
        """
            state, gradient: [batch, state_dim], gamma: [batch, state_dim, state_dim]
            output: h_min [batch], u_star [batch, control_dim]
            The minimum is approximate, using a grid or a fixed number of Adam steps.
        """
        if state.ndim != 2:
            raise ValueError("minimise expects [batch,state_dim]")

        # Scalar control: evaluate a grid including both endpoints, then choose its minimum.
        # Detach the search inputs; re-evaluate H at the selected control for outer training.
        if self.model_config.control_dim == 1:
            batch = state.shape[0]
            lo = float(self.problem.control_low.item())
            hi = float(self.problem.control_high.item())
            K = max(81, 20 * self.optimizer_config.num_starts + 1) # number of scalar grid points
            grid = torch.linspace(
                lo, hi, K, dtype=state.dtype, device=state.device
            ) # [K]
            controls = grid[None, :, None].expand(batch, -1, -1) # [batch, K, 1]
            s = state.detach()[:, None, :].expand(-1, K, -1) # [batch, K, state_dim]
            z = gradient.detach()[:, None, :].expand(-1, K, -1) # [batch, K, state_dim]
            g = gamma.detach()[:, None, :, :].expand(-1, K, -1, -1) # [batch, K, state_dim, state_dim]
            with torch.no_grad():
                values = self.evaluate(s, z, g, controls) # [batch, K]
                idx = values.argmin(dim=1) # [batch], best grid point for each state
                bidx = torch.arange(batch, device=state.device)
                u_star = controls[bidx, idx].detach() # [batch, 1]
            h_min = self.evaluate(state, gradient, gamma, u_star) # original inputs retain gradients, u_star is treated as fixed
            return h_min, u_star

        # Multidimensional control: run projected Adam from K starts for each state.
        K = self.optimizer_config.num_starts
        batch = state.shape[0]
        starts = self._sobol_starts(
            K, dtype=state.dtype, device=state.device
        )[None, :, :].expand(batch, -1, -1).clone() # [batch, K, control_dim], same initial starts for each state
        variable = nn.Parameter(starts) # controls to optimise in the inner search
        inner = optim.Adam([variable], lr=self.optimizer_config.learning_rate)
        s = state.detach()[:, None, :].expand(-1, K, -1) # [batch, K, state_dim], fixed during the inner search
        z = gradient.detach()[:, None, :].expand(-1, K, -1) # [batch, K, state_dim]
        g = gamma.detach()[:, None, :, :].expand(-1, K, -1, -1) # [batch, K, state_dim, state_dim]
        with torch.enable_grad(): # the inner control search also needs gradients during evaluation
            for _ in range(self.optimizer_config.num_steps):
                u = self.project_control(variable) # [batch, K, control_dim]
                values = self.evaluate(s, z, g, u) # [batch, K]
                loss = values.mean() # scalar objective over all states and starts
                inner.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [variable], self.optimizer_config.gradient_clip
                ) # clip the total control-gradient norm
                inner.step()
                with torch.no_grad():
                    variable.copy_(self.project_control(variable)) # project back into the box after each update
        with torch.no_grad():
            candidates = self.project_control(variable) # [batch, K, control_dim], final candidates
            values = self.evaluate(s, z, g, candidates) # [batch, K]
            idx = values.argmin(dim=1) # [batch], best final candidate for each state
            bidx = torch.arange(batch, device=state.device)
            u_star = candidates[bidx, idx].detach() # [batch, control_dim], no derivative through the chosen control
        h_min = self.evaluate(state, gradient, gamma, u_star) # re-evaluate with original inputs, treating u_star as fixed
        return h_min, u_star
