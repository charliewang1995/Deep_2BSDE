"""Fresh-path evaluation."""
from dataclasses import dataclass
import math
import torch


@dataclass
class EvaluationResult:
    paths: dict # terminal values and the full paths returned by the model
    diagnostics: dict # scalar summaries of posterior corrections and selected controls


def evaluate(model, *, num_paths=1000):
    """
        Simulate fresh paths under the fixed auxiliary control.
        output: EvaluationResult, containing paths and diagnostics
    """
    cfg = model.config
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    # fresh independent Brownian increments, each component has mean zero and variance dt
    dI = math.sqrt(cfg.dt) * torch.randn(
        num_paths, cfg.num_time_steps, cfg.observation_dim,
        dtype=dtype, device=device
    ) # [num_paths, num_time_steps, observation_dim]
    with torch.no_grad(): # no outer gradient graph; the inner control optimiser can still enable gradients
        paths = model(dI, return_paths=True) # state paths follow u_aux; u_star is computed along these paths

    # u_star_path: [num_paths, num_time_steps, control_dim]
    diagnostics = {
        "posterior_correction_rate":
            float(paths["posterior_correction_rate"].detach().cpu()), # fraction flagged invalid before correction, averaged over paths and time steps
        "mean_u_star":
            float(paths["u_star_path"].mean().detach().cpu()), # average over paths, time steps and control coordinates
    }
    return EvaluationResult(paths=paths, diagnostics=diagnostics)


def evaluate_time_blocks(
    model, *, num_paths=1000, initial_states=None, start_step=0,
    evaluate_policy=True,
):
    """
        initial_states: optional [state_dim] or [num_paths, state_dim], otherwise use the model's sampler
        start_step: global fine-grid starting index; simulate the remaining horizon
        output: EvaluationResult with auxiliary paths and, optionally, independent policy paths

        The main rollout uses each path's own minimiser based on propagated Z and
        the Gamma network. Its Hamiltonian gap is zero up to numerical roundoff.
        The separate policy diagnostic uses fresh noise and value-network derivatives
        at the same frozen starting states. Neither cost is a training target.
    """
    cfg = model.config
    if isinstance(num_paths, bool) or not isinstance(num_paths, int) or num_paths < 1:
        raise ValueError("num_paths must be a positive integer")
    if isinstance(start_step, bool) or not isinstance(start_step, int) or not 0 <= start_step <= cfg.num_time_steps:
        raise ValueError("start_step must be an integer between zero and num_time_steps")
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad(): # the model enables state derivatives internally where needed
            dI = math.sqrt(cfg.dt) * torch.randn(
                num_paths, cfg.num_time_steps - start_step, cfg.observation_dim,
                dtype=dtype, device=device,
            ) # [num_paths, remaining_steps, observation_dim]
            paths = model(
                dI, return_paths=True, initial_states=initial_states,
                start_step=start_step, return_training=True,
            )
            terminal_value = model.problem.terminal_cost(paths["S_T"])
            terminal_gradient = model.geometry.project_gradient(
                model.problem.terminal_gradient(paths["S_T"])
            )
            corrected_cost = paths["raw_auxiliary_cost"] - paths["gap_integral"] # [num_paths]
            # Discrete algebra: raw_cost-gap = initial_value + g(S_T)-R_T + martingale_integral.
            identity_residual = corrected_cost - (
                paths["initial_values"] + terminal_value - paths["R_T"]
                + paths["martingale_integral"]
            ) # close to zero by construction; this alone does not certify value accuracy
            metrics = {
                "posterior_correction_rate": paths["posterior_correction_rate"],
                "terminal_value_loss": (paths["R_T"] - terminal_value).square().mean(),
                "terminal_gradient_loss": (paths["Z_T"] - terminal_gradient).square().mean(),
                "consistency_loss": paths["consistency_loss"],
                "boundary_loss": paths["boundary_loss"],
                "gradient_consistency_loss": paths["gradient_consistency_loss"],
                "mean_predicted_value": paths["initial_values"].mean(),
                "mean_auxiliary_cost": paths["raw_auxiliary_cost"].mean(),
                "mean_gap": paths["gap_integral"].mean(),
                "mean_corrected_auxiliary_cost": corrected_cost.mean(),
                "mean_auxiliary_martingale": paths["martingale_integral"].mean(),
                "auxiliary_identity_max_abs_residual": identity_residual.abs().max(),
            }
            diagnostics = {
                name: float(value.detach().cpu()) for name, value in metrics.items()
            } # summaries over the pathwise rollout, retaining the existing diagnostic names
            paths["corrected_auxiliary_cost"] = corrected_cost
            paths["auxiliary_identity_residual"] = identity_residual

            if evaluate_policy:
                frozen_starts = paths["S_path"][:, 0, :].detach().clone() # reuse exactly the sampled auxiliary starting states
                policy_dI = math.sqrt(cfg.dt) * torch.randn(
                    num_paths, cfg.num_time_steps - start_step, cfg.observation_dim,
                    dtype=dtype, device=device,
                ) # independent of the auxiliary increments
                policy = model.rollout_policy(
                    policy_dI, initial_states=frozen_starts, start_step=start_step,
                )
                policy_error = policy["policy_cost"] - policy["initial_values"] # paired per-path cost minus prediction
                diagnostics.update({
                    "mean_policy_cost": float(policy["policy_cost"].mean().detach().cpu()),
                    "mean_policy_value_error": float(policy_error.mean().detach().cpu()),
                    "policy_posterior_correction_rate": float(policy["posterior_correction_rate"].detach().cpu()),
                }) # a policy cost need not lie above an inaccurate learned value prediction
                policy["value_error"] = policy_error
                paths["policy"] = policy
    finally:
        model.train(was_training)

    return EvaluationResult(paths=paths, diagnostics=diagnostics)
