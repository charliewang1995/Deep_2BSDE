"""Geometric 2BSDE models with fixed-auxiliary or pathwise Hamiltonian controls."""
import torch
from torch import nn
import math
import warnings
from .dynamics import state_dynamics, stabilise_state
from .networks import GeometricParallelBlock, TimeGeometricBlock, make_mlp


class GeometricClassicalFixedAux2BSDE(nn.Module):
    def __init__(
        self, *, problem, config, geometry, hamiltonian_optimizer
    ):
        super().__init__()
        self.problem = problem
        self.config = config
        self.geometry = geometry
        self.hamiltonian_optimizer = hamiltonian_optimizer

        if getattr(problem, "auxiliary_control_path", None) is None:
            raise ValueError("The fixed-auxiliary model requires problem.auxiliary_control_path")

        # one separate network block for A and Gamma at each time after t=0,
        # block k is evaluated at S_{k+1} and used for the next time step
        self.blocks = nn.ModuleList(
            GeometricParallelBlock(config, geometry)
            for _ in range(max(config.num_time_steps - 1, 0))
        )

        self.initial_value = nn.Parameter(
            torch.tensor(float(config.value_initial_guess))
        ) # scalar, trainable initial value R0 = V0 for the fixed initial state
        self.initial_gradient_raw = nn.Parameter(
            torch.zeros(config.reduced_vector_dim)
        ) # [reduced_vector_dim], observation and posterior tangent coordinates of Z0
        self.initial_drift_gradient_raw = nn.Parameter(
            torch.zeros(config.reduced_vector_dim) #A0
        ) # [reduced_vector_dim], coordinates of the drift A0 of Z
        self.initial_gamma_raw = nn.Parameter(
            config.gamma_initial_scale
            * torch.randn(config.gamma_parameter_dim)
        ) # [gamma_parameter_dim], independent coefficients of the reduced symmetric Gamma0

    def initial_gradient(self):
        return self.geometry.vector_from_raw(self.initial_gradient_raw) # [state_dim], posterior part sums to zero

    def initial_drift_gradient(self):
        return self.geometry.vector_from_raw(self.initial_drift_gradient_raw) # [state_dim], posterior part sums to zero

    def initial_gamma(self):
        return self.geometry.gamma_from_raw(self.initial_gamma_raw) # [state_dim, state_dim], symmetric with tangent posterior blocks

    def forward(self, innovations, return_paths=False):
        """
            innovations: [batch, num_time_steps, observation_dim], Brownian increments dI, already scaled by sqrt(dt)
            R: value process, Z: state gradient, A: drift of Z, Gamma: Hessian approximation
            output: terminal R, Z, S and correction rate, or a dictionary including all paths
        """
        cfg = self.config
        if innovations.ndim != 3:
            raise ValueError("innovations must be [batch,time,observation_dim]")
        if innovations.shape[1:] != (
            cfg.num_time_steps, cfg.observation_dim
        ):
            raise ValueError("innovation shape mismatch", 
                             f"got {innovations.shape[1:]}", 
                             f"expected {(cfg.num_time_steps, cfg.observation_dim)}.")

        batch = innovations.shape[0]
        dtype, device = innovations.dtype, innovations.device

        y0 = self.problem.initial_observation.to(dtype=dtype, device=device) # [observation_dim]
        p0 = self.problem.initial_posterior.to(dtype=dtype, device=device) # [num_hidden_states]
        S0 = torch.cat((y0, p0)) # [state_dim], separated state S = (Y, p)
        state = S0[None, :].expand(batch, -1).clone() # [batch, state_dim], same initial state on every path

        R = self.initial_value.to(dtype=dtype, device=device).expand(batch) # [batch]
        Z = self.initial_gradient()[None, :].expand(batch, -1) # [batch, state_dim]
        A = self.initial_drift_gradient()[None, :].expand(batch, -1) # [batch, state_dim]
        Gamma = self.initial_gamma()[None, :, :].expand(batch, -1, -1) # [batch, state_dim, state_dim]

        S_path = [state]
        R_path = [R]
        Z_path = [Z]
        A_path, G_path, u_aux_path, u_star_path = [], [], [], []
        correction_rates = []

        aux_path = self.problem.auxiliary_control_path.to(dtype=dtype, device=device) # [num_time_steps, control_dim]

        for step in range(cfg.num_time_steps):
            time = step * cfg.dt
            u_aux = aux_path[step][None, :].expand(batch, -1) # [batch, control_dim], fixed auxiliary control for this step

            drift_aux, sigma_aux, _ = state_dynamics(
                state, u_aux,
                problem=self.problem, config=cfg, geometry=self.geometry
            ) # drift_aux [batch, state_dim], sigma_aux [batch, state_dim, observation_dim]

            h_min, u_star = self.hamiltonian_optimizer.minimise(
                state, 
                Z, 
                Gamma,
                time=time,
            ) # h_min [batch], u_star [batch, control_dim], approximate Hamiltonian minimiser at the current auxiliary state

            # sigma_aux, shape [batch, state_dim, observation_dim]
            # Gamma,     shape [batch, state_dim, state_dim]
            aux_quad = torch.einsum(
                "bia,bij,bja->b", sigma_aux, Gamma, sigma_aux
            ) # [batch], trace(sigma_aux.T Gamma sigma_aux)
            driver = (
                -h_min
                + (drift_aux * Z).sum(dim=-1)
                + 0.5 * aux_quad
            ) # [batch], drift of R under the auxiliary dynamics
            Z_sigma = torch.einsum("bi,bia->ba", Z, sigma_aux) # [batch, observation_dim], Z.T sigma_aux

            dI = innovations[:, step, :] # [batch, observation_dim], same increment for R, Z and S
            R = R + driver * cfg.dt + (Z_sigma * dI).sum(dim=-1) # [batch], Euler step for the value process

            # dZ = A dt + Gamma sigma_aux dI,
            # A is the drift of Z, not the drift of the state S
            Z = (
                Z
                + A * cfg.dt
                + torch.einsum(
                    "bij,bja,ba->bi", Gamma, sigma_aux, dI
                ) # [batch, state_dim]
            )
            Z = self.geometry.project_gradient(Z) # keep the posterior part zero-sum

            # the state follows u_aux; u_star enters the Hamiltonian above
            state = (
                state
                + drift_aux * cfg.dt
                + torch.einsum("bia,ba->bi", sigma_aux, dI) # [batch, state_dim]
            )
            state, corr = stabilise_state(state, config=cfg) # clip and renormalise posterior; corr is the fraction flagged invalid before correction

            # A, Gamma and controls belong to the start of this step;
            # S, R and Z below are the updated values at the end of this step
            A_path.append(A)
            G_path.append(Gamma)
            u_aux_path.append(u_aux)
            u_star_path.append(u_star)
            correction_rates.append(corr)
            S_path.append(state)
            R_path.append(R)
            Z_path.append(Z)

            if step < cfg.num_time_steps - 1:
                A, Gamma = self.blocks[step](state) # evaluate at the new state, for use in the next step

        correction_rate = torch.stack(correction_rates).mean() # scalar, invalid-posterior fraction averaged over time steps

        if not return_paths:
            return R, Z, state, correction_rate # [batch], [batch, state_dim], [batch, state_dim], scalar

        return {
            "R_T": R, # [batch]
            "Z_T": Z, # [batch, state_dim]
            "S_T": state, # [batch, state_dim]
            "R_path": torch.stack(R_path, dim=1), # [batch, num_time_steps+1], includes t=0 and T
            "Z_path": torch.stack(Z_path, dim=1), # [batch, num_time_steps+1, state_dim]
            "S_path": torch.stack(S_path, dim=1), # [batch, num_time_steps+1, state_dim]
            "A_path": torch.stack(A_path, dim=1), # [batch, num_time_steps, state_dim]
            "Gamma_path": torch.stack(G_path, dim=1), # [batch, num_time_steps, state_dim, state_dim]
            "u_aux_path": torch.stack(u_aux_path, dim=1), # [batch, num_time_steps, control_dim]
            "u_star_path": torch.stack(u_star_path, dim=1), # [batch, num_time_steps, control_dim], minimisers along auxiliary paths
            "posterior_correction_rate": correction_rate, # scalar
        }

    def save_blocks(self, path): # save the A and Gamma networks; initial trainable parameters are not included
        torch.save(self.blocks.state_dict(), path)

    def load_blocks(self, path, *, map_location="cpu"): # restore the network blocks into a matching architecture
        self.blocks.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True)
        )


class GeometricTimeBlockAdaptive2BSDE(nn.Module):
    """
        State-dependent V0, shared time blocks and pathwise Hamiltonian controls.

        Every process and Hamiltonian is updated on the original fine time grid.
        Each block reuses its network parameters, evaluated at the current state.
        At each fine step, each path uses its own Hamiltonian minimiser to update
        its state and 2BSDE processes. No controls are pooled across paths.

        num_blocks defaults to ceil(num_time_steps/10). Blocks share network
        parameters over time; the controls are recomputed at every fine step.
        Legacy first_control_fraction/last_control_fraction arguments are ignored.

        initial_state_sampler(batch_size, *, dtype, device) may return [batch, state_dim].
        This lets the existing training loop learn V0 across different initial states.
        Without a sampler or explicit initial_states, all paths start at the problem's
        initial state. Training at that single state does not determine V0 elsewhere.

        Controls use the propagated Z and the current block's Gamma approximation.
        The A network receives each path's own control. Saving model.state_dict()
        includes all value/A/Gamma networks and the initial reference state;
        save_blocks() saves all value/A/Gamma network weights.
    """

    def __init__(
        self, *, 
        problem, 
        config, 
        geometry, 
        hamiltonian_optimizer,
        num_blocks=None, 
        first_control_fraction=None, last_control_fraction=None,
        initial_state_sampler=None, # optional, used when initial state is not a fixed value 
        track_gradient_consistency=False, # optional, to keep track of whether autograd of V is close to Z 
    ):
        super().__init__()
        if config.num_time_steps < 1 or config.horizon <= 0:
            raise ValueError("A positive horizon and positive integer num_time_steps are required")
        
        if num_blocks is None:
            num_blocks = max(1, math.ceil(config.num_time_steps / 10))

        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int):
            raise ValueError("num_blocks must be an integer")
        
        if not 1 <= num_blocks <= config.num_time_steps:
            raise ValueError("num_blocks must be between 1 and num_time_steps")
        
        if first_control_fraction is not None or last_control_fraction is not None:
            warnings.warn(
                "Control fractions are ignored: each path now uses its own Hamiltonian minimiser.",
                DeprecationWarning, stacklevel=2,
            )
        
        if initial_state_sampler is not None and not callable(initial_state_sampler):
            raise TypeError("initial_state_sampler must be callable")
        
        self.problem = problem
        self.config = config
        self.geometry = geometry
        self.hamiltonian_optimizer = hamiltonian_optimizer
        self.num_blocks = num_blocks

        self.initial_state_sampler = initial_state_sampler
        self.track_gradient_consistency = bool(track_gradient_consistency)

        self.block_for_step = tuple(
            step * num_blocks // config.num_time_steps for step in range(config.num_time_steps)
        )
        self.block_start_steps = tuple(
            (j * config.num_time_steps + num_blocks - 1) // num_blocks for j in range(num_blocks)
        ) # actual block boundaries on the fine grid, including non-divisible grids

        self.blocks = nn.ModuleList(TimeGeometricBlock(config, geometry) for _ in range(num_blocks))

        default_state = torch.cat((problem.initial_observation, problem.initial_posterior)).detach()
        if default_state.shape != (config.state_dim,):
            raise ValueError("The default initial state must have shape [state_dim]")
        self.register_buffer("default_initial_state", default_state.clone()) # starting data, not a value anchor

    def _time_tensor(self, time, state):
        if state.ndim < 1 or state.shape[-1] != self.config.state_dim:
            raise ValueError("state must have shape [..., state_dim]")
        t = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        if t.ndim == 0:
            t = t.expand(state.shape[:-1])
        if t.shape != state.shape[:-1]:
            raise ValueError("time must be scalar or have shape state.shape[:-1]")
        if (t < 0).any() or (t > self.config.horizon).any():
            raise ValueError("time must be between zero and the horizon")
        return t

    def value(self, time, state, *, block_index=None):
        """
            time: scalar or [...]; state: [..., state_dim]; output: [...]
            Select the block by time, taking the right block at a boundary and the last at T.
            block_index can explicitly select either adjacent head for a boundary check.
        """
        t = self._time_tensor(time, state)

        if block_index is not None:
            if isinstance(block_index, bool) or not isinstance(block_index, int) or not 0 <= block_index < self.num_blocks:
                raise ValueError("block_index must be a valid integer block index")
            return self.blocks[block_index].value(t, state)
        
        boundaries = torch.tensor(
            [k * self.config.dt for k in self.block_start_steps[1:]],
            dtype=state.dtype, device=state.device,
        )
        indices = torch.bucketize(t.contiguous(), boundaries, right=True)
        flat_state, flat_time = state.reshape(-1, self.config.state_dim), t.reshape(-1)
        output = torch.empty(flat_time.shape, dtype=state.dtype, device=state.device)
        for j, block in enumerate(self.blocks):
            mask = indices.reshape(-1) == j
            if mask.any():
                output[mask] = block.value(flat_time[mask], flat_state[mask])
        return output.reshape(state.shape[:-1]) # indexed assignment retains gradients to the selected heads

    def value_and_gradient(self, time, state, *, block_index=None):
        """
            output: value [batch], projected state gradient [batch, state_dim]
            Used at each restart and, optionally, for intermediate gradient consistency.
        """
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            s = state if state.requires_grad else state.detach().clone().requires_grad_(True)
            value = self.value(time, s, block_index=block_index)
            gradient = torch.autograd.grad(value.sum(), s, create_graph=outer_grad)[0]
            gradient = self.geometry.project_gradient(gradient)
        return (value, gradient) if outer_grad else (value.detach(), gradient.detach())

    def _initial_states(self, initial_states, batch, *, dtype, device):
        if initial_states is None and self.initial_state_sampler is not None:
            initial_states = self.initial_state_sampler(batch, dtype=dtype, device=device)
        if initial_states is None:
            initial_states = self.default_initial_state
        state = torch.as_tensor(initial_states, dtype=dtype, device=device)
        if state.shape == (self.config.state_dim,):
            state = state[None, :].expand(batch, -1)
        if state.shape != (batch, self.config.state_dim):
            raise ValueError("initial_states must have shape [state_dim] or [batch, state_dim]")
        p = state[..., self.config.observation_dim:]
        if (not torch.isfinite(state).all() or (p < 0).any()
                or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-6, rtol=0)):
            raise ValueError("Initial states must be finite, with nonnegative posteriors summing to one")
        return state.clone()

    def _check_innovations(self, innovations, start_step):
        if isinstance(start_step, bool) or not isinstance(start_step, int) or not 0 <= start_step <= self.config.num_time_steps:
            raise ValueError("start_step must be an integer between 0 and num_time_steps")
        expected = (self.config.num_time_steps - start_step, self.config.observation_dim)
        if innovations.ndim != 3 or innovations.shape[1:] != expected or innovations.shape[0] == 0:
            raise ValueError(f"innovations must have shape [nonempty_batch, {expected[0]}, {expected[1]}]")
        parameter = next(self.parameters())
        if innovations.dtype != parameter.dtype or innovations.device != parameter.device:
            raise ValueError("innovations must have the same dtype and device as the model")
        if self.geometry.basis.dtype != parameter.dtype or self.geometry.basis.device != parameter.device:
            raise ValueError("geometry.basis must have the same dtype and device as the model")

    def forward(
        self, innovations, return_paths=False, *, initial_states=None,
        start_step=0, return_training=False,
    ):
        """
            innovations: [batch, num_time_steps-start_step, observation_dim]
            All increments have variance dt; the global time grid is not rescaled on restart.
            output: terminal tuple, or training dictionary, or dictionary with all fine-grid paths.
        """
        self._check_innovations(innovations, start_step)
        cfg = self.config
        batch, remaining, _ = innovations.shape
        state = self._initial_states(initial_states, batch, dtype=innovations.dtype, device=innovations.device)
        start_time = start_step * cfg.dt if start_step < cfg.num_time_steps else cfg.horizon
        initial_block = self.block_for_step[start_step] if remaining else self.num_blocks - 1
        R, Z = self.value_and_gradient(start_time, state, block_index=initial_block)
        initial_values = R
        zero = R.sum() * 0.0
        consistency_terms, boundary_terms, gradient_terms = [], [], []
        correction_rates = []
        metrics = return_paths or return_training
        running_sum = torch.zeros(batch, dtype=state.dtype, device=state.device)
        gap_sum, martingale_sum = torch.zeros_like(running_sum), torch.zeros_like(running_sum)
        if return_paths:
            S_path, R_path, Z_path, value_path = [state], [R], [Z], [R]
            A_path, G_path, u_aux_path, u_star_path = [], [], [], []

        for local_step, step in enumerate(range(start_step, cfg.num_time_steps)):
            time, block_index = step * cfg.dt, self.block_for_step[step]
            block = self.blocks[block_index]
            Gamma = block.gamma(time, state) # evaluate the current block at every fine step
            h_min, u_star = self.hamiltonian_optimizer.minimise(state, Z, Gamma, time=time)
            if u_star.shape != (batch, cfg.control_dim):
                raise ValueError("Hamiltonian minimisers must have shape [batch, control_dim]")
            u_aux = u_star.detach() # each path uses its own fine-step minimiser; no batch averaging
            A = block.drift_gradient(time, state, u_aux) # the Itô drift of Z also depends on the actual auxiliary control
            drift, sigma, _ = state_dynamics(
                state, u_aux, problem=self.problem, config=cfg, geometry=self.geometry
            )
            linear = (drift * Z).sum(-1)
            quadratic = 0.5 * torch.einsum("bia,bij,bja->b", sigma, Gamma, sigma)
            dI = innovations[:, local_step, :]
            martingale_increment = (torch.einsum("bi,bia->ba", Z, sigma) * dI).sum(-1)
            if metrics:
                with torch.no_grad(): # cost and gap diagnostics never enter the optimisation graph
                    running = self.problem.running_cost(state, u_aux, time=time)
                    if running.shape != (batch,):
                        raise ValueError("running_cost must return [batch] for a single time step")
                    running_sum = running_sum + running * cfg.dt
                    gap_sum = gap_sum + (running + linear.detach() + quadratic.detach() - h_min.detach()) * cfg.dt
                    martingale_sum = martingale_sum + martingale_increment.detach()

            R = R + (-h_min + linear + quadratic) * cfg.dt + martingale_increment
            Z = self.geometry.project_gradient(
                Z + A * cfg.dt + torch.einsum("bij,bja,ba->bi", Gamma, sigma, dI)
            )
            state = state + drift * cfg.dt + torch.einsum("bia,ba->bi", sigma, dI)
            state, correction = stabilise_state(state, config=cfg)
            correction_rates.append(correction)
            next_time = (step + 1) * cfg.dt if step + 1 < cfg.num_time_steps else cfg.horizon
            next_block = self.block_for_step[step + 1] if step + 1 < cfg.num_time_steps else self.num_blocks - 1
            if metrics:
                if self.track_gradient_consistency:
                    network_value, network_gradient = self.value_and_gradient(next_time, state, block_index=next_block)
                    gradient_terms.append((network_gradient - Z).square().mean())
                else:
                    network_value = self.blocks[next_block].value(next_time, state)
                consistency_terms.append((network_value - R).square().mean()) # both approximations learn; R is never overwritten
                if next_block != block_index:
                    left_value = block.value(next_time, state)
                    boundary_terms.append((left_value - network_value).square().mean())
            if return_paths:
                S_path.append(state)
                R_path.append(R)
                Z_path.append(Z)
                value_path.append(network_value)
                A_path.append(A)
                G_path.append(Gamma)
                u_aux_path.append(u_aux)
                u_star_path.append(u_star)

        correction_rate = torch.stack(correction_rates).mean() if correction_rates else zero.detach()
        if not metrics:
            return R, Z, state, correction_rate
        with torch.no_grad():
            terminal_cost = self.problem.terminal_cost(state)
            if terminal_cost.shape != (batch,):
                raise ValueError("terminal_cost must return [batch]")
            raw_cost = running_sum + terminal_cost
        result = {
            "R_T": R, "Z_T": Z, "S_T": state,
            "initial_values": initial_values,
            "consistency_loss": torch.stack(consistency_terms).mean() if consistency_terms else zero,
            "boundary_loss": torch.stack(boundary_terms).mean() if boundary_terms else zero,
            "gradient_consistency_loss": torch.stack(gradient_terms).mean() if gradient_terms else zero,
            "raw_auxiliary_cost": raw_cost,
            "gap_integral": gap_sum,
            "martingale_integral": martingale_sum,
            "posterior_correction_rate": correction_rate,
            "start_step": start_step,
        }
        if return_paths:
            def stack_or_empty(items, tail):
                return torch.stack(items, dim=1) if items else state.new_empty((batch, 0) + tail)
            result.update({
                "S_path": torch.stack(S_path, 1), # [batch, remaining+1, state_dim]
                "R_path": torch.stack(R_path, 1),
                "Z_path": torch.stack(Z_path, 1),
                "value_path": torch.stack(value_path, 1), # independently evaluated value heads, alongside propagated R
                "A_path": stack_or_empty(A_path, (cfg.state_dim,)),
                "Gamma_path": stack_or_empty(G_path, (cfg.state_dim, cfg.state_dim)),
                "u_aux_path": stack_or_empty(u_aux_path, (cfg.control_dim,)), # same pathwise controls as u_star_path
                "u_star_path": stack_or_empty(u_star_path, (cfg.control_dim,)),
                "time_path": torch.tensor(
                    [k * cfg.dt for k in range(start_step, cfg.num_time_steps)] + [cfg.horizon],
                    dtype=state.dtype, device=state.device,
                ),
                "block_index_path": torch.tensor(self.block_for_step[start_step:], dtype=torch.long, device=state.device),
            })
        return result

    def value_gradient_hessian(self, time, state):
        """
            Evaluation helper: derivatives of the scalar value network at the CURRENT state.
            output: detached V [batch], projected Z [batch,D], projected Hessian [batch,D,D].
            A full Hessian is formed, so policy diagnostics can be expensive for large D.
        """
        with torch.enable_grad():
            s = state.detach().clone().requires_grad_(True)
            value = self.value(time, s)
            gradient = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
            rows = []
            for i in range(self.config.state_dim):
                component = gradient[..., i].sum()
                row = torch.autograd.grad(component, s, retain_graph=True, allow_unused=True)[0] if component.requires_grad else None
                rows.append(torch.zeros_like(s) if row is None else row)
            hessian = torch.stack(rows, dim=-2)
            Z = self.geometry.project_gradient(gradient)
            Gamma = self.geometry.project_hessian(hessian)
        return value.detach(), Z.detach(), Gamma.detach()

    def rollout_policy(self, innovations, *, initial_states=None, start_step=0):
        """
            Fresh individual feedback rollout: minimise H using derivatives of v_j(t,S).
            Controls are recomputed using value-network derivatives at these policy states.
            The training rollout instead uses propagated Z and the separate Gamma network.
            Costs are diagnostic only, with no outer graph.
        """
        self._check_innovations(innovations, start_step)
        cfg = self.config
        batch = innovations.shape[0]
        with torch.no_grad():
            state = self._initial_states(initial_states, batch, dtype=innovations.dtype, device=innovations.device)
            start_time = start_step * cfg.dt if start_step < cfg.num_time_steps else cfg.horizon
            initial_values = self.value(start_time, state)
            states, controls, corrections = [state], [], []
            cost = torch.zeros_like(initial_values)
            for local_step, step in enumerate(range(start_step, cfg.num_time_steps)):
                time = step * cfg.dt
                _, Z, Gamma = self.value_gradient_hessian(time, state)
                _, control = self.hamiltonian_optimizer.minimise(state, Z, Gamma, time=time)
                cost = cost + self.problem.running_cost(state, control, time=time) * cfg.dt
                drift, sigma, _ = state_dynamics(
                    state, control, problem=self.problem, config=cfg, geometry=self.geometry
                )
                state = state + drift * cfg.dt + torch.einsum("bia,ba->bi", sigma, innovations[:, local_step, :])
                state, correction = stabilise_state(state, config=cfg)
                states.append(state)
                controls.append(control)
                corrections.append(correction)
            cost = cost + self.problem.terminal_cost(state)
            return {
                "policy_cost": cost, "initial_values": initial_values,
                "S_path": torch.stack(states, dim=1),
                "u_path": torch.stack(controls, dim=1) if controls else state.new_empty((batch, 0, cfg.control_dim)),
                "posterior_correction_rate": torch.stack(corrections).mean() if corrections else state.new_zeros(()),
            }

    def save_blocks(self, path): # all value/A/Gamma network weights; constructor settings are supplied again on load
        torch.save(self.blocks.state_dict(), path)

    def load_blocks(self, path, *, map_location="cpu"):
        self.blocks.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
