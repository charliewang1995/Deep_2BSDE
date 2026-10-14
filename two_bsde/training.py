"""Outer 2BSDE training loop."""
from dataclasses import dataclass, field
import math
import torch


@dataclass
class TrainingHistory:
    total_loss: list = field(default_factory=list) # weighted total loss at each epoch
    value_loss: list = field(default_factory=list) # terminal value loss at each epoch
    gradient_loss: list = field(default_factory=list) # terminal gradient loss at each epoch


def train(model, *, model_config, training_config, verbose=True):
    """
        Train the initial values and neural networks using terminal conditions.
        output: TrainingHistory, containing one loss value per epoch
    """
    history = TrainingHistory()
    other_parameters = [p for p in model.parameters() if p is not model.initial_value] # all trainable parameters except V0
    optimizer = torch.optim.Adam(
    [
        {
            "params": [model.initial_value],
            "lr": training_config.initial_value_lr, # separate learning rate for the initial value V0
        },
        {
            "params": other_parameters,
            "lr": training_config.learning_rate, # learning rate for initial Z, A, Gamma and the networks
        },
    ]
    )

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    report_every = max(training_config.epochs // 10, 1) # print progress about ten times during training

    for epoch in range(1, training_config.epochs + 1):
        # fresh independent Brownian increments, each component has mean zero and variance dt
        dI = math.sqrt(model_config.dt) * torch.randn(
            training_config.batch_size,
            model_config.num_time_steps,
            model_config.observation_dim,
            dtype=dtype,
            device=device,
        ) # [batch, num_time_steps, observation_dim]

        # R_T: terminal value approximation [batch], Z_T: terminal state gradient approximation [batch, state_dim]
        # S_T: terminal state (Y_T, p_T) [batch, state_dim], corr: fraction flagged invalid before correction, averaged over time
        R_T, Z_T, S_T, corr = model(dI, return_paths=False)

        target_value = model.problem.terminal_cost(S_T) # g(S_T), [batch]
        target_gradient = model.geometry.project_gradient(
            model.problem.terminal_gradient(S_T)
        ) # [batch, state_dim], project the posterior part of Dg(S_T) onto the simplex tangent space

        value_loss = (R_T - target_value).square().mean() # mean squared terminal value error over the batch
        gradient_loss = (Z_T - target_gradient).square().mean() # mean squared terminal gradient error over the batch and all state coordinates
        total_loss = (
            training_config.value_loss_weight * value_loss
            + training_config.gradient_loss_weight * gradient_loss
        ) # weighted sum of the two terminal losses

        optimizer.zero_grad(set_to_none=True) # clear gradients from the previous epoch
        total_loss.backward() # differentiate the terminal loss through the 2BSDE updates
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.gradient_clip
        ) # bound the norm of the combined parameter gradient
        optimizer.step() # update the initial values and network parameters

        history.total_loss.append(float(total_loss.detach().cpu())) # store scalar losses without the gradient graph
        history.value_loss.append(float(value_loss.detach().cpu()))
        history.gradient_loss.append(float(gradient_loss.detach().cpu()))

        if verbose and (epoch == 1 or epoch % report_every == 0):
            print(
                f"epoch={epoch:4d}  total_loss={history.total_loss[-1]:.3e}, where  "
                f"value_loss={history.value_loss[-1]:.3e}  and "
                f"gradient_loss={history.gradient_loss[-1]:.3e}  "
                f"V0={model.initial_value.item():.6f}  corr_rate={corr.item():.2e}"
            )

    return history


@dataclass
class TimeBlockTrainingHistory:
    total_loss: list = field(default_factory=list) # weighted training loss at each epoch
    terminal_value_loss: list = field(default_factory=list) # mean squared error R_T - g(S_T)
    terminal_gradient_loss: list = field(default_factory=list) # mean squared error in the projected terminal gradient
    consistency_loss: list = field(default_factory=list) # value-head agreement with the propagated R process
    boundary_loss: list = field(default_factory=list) # agreement of adjacent value heads at block transitions
    gradient_consistency_loss: list = field(default_factory=list) # optional agreement of value derivatives with propagated Z
    mean_initial_value: list = field(default_factory=list) # mean value prediction at this epoch's starting time
    start_step: list = field(default_factory=list) # global fine-grid starting index for each epoch
    mean_auxiliary_cost: list = field(default_factory=list) # diagnostic only; not a training target
    mean_gap: list = field(default_factory=list) # mean integrated Hamiltonian gap along auxiliary paths


def train_time_blocks(
    model, *, training_config, consistency_weight=1.0, boundary_weight=0.1,
    gradient_consistency_weight=0.0, sample_start_times=True,
    initial_start_probability=0.5, verbose=True,
):
    """
        Train shared time-block value, A and Gamma networks using remaining-horizon paths.
        Each epoch has one common global start_step; states are drawn by the model's sampler.
        output: TimeBlockTrainingHistory, containing one entry per epoch
    """
    weights = {
        "value_loss_weight": training_config.value_loss_weight,
        "gradient_loss_weight": training_config.gradient_loss_weight,
        "consistency_weight": consistency_weight,
        "boundary_weight": boundary_weight,
        "gradient_consistency_weight": gradient_consistency_weight,
    }
    for name, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(initial_start_probability) or not 0 <= initial_start_probability <= 1:
        raise ValueError("initial_start_probability must lie in [0, 1]")
    if gradient_consistency_weight > 0 and not model.track_gradient_consistency:
        raise ValueError("Enable track_gradient_consistency on the model before using its loss")
    if training_config.epochs < 0 or training_config.batch_size < 1:
        raise ValueError("epochs must be nonnegative and batch_size must be positive")

    cfg = model.config
    history = TimeBlockTrainingHistory()
    value_parameters = [
        parameter for block in model.blocks for parameter in block.value_net.parameters()
        if parameter.requires_grad
    ] # all scalar value heads, including those in later time blocks
    value_parameter_ids = {id(parameter) for parameter in value_parameters}
    other_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in value_parameter_ids
    ] # A and Gamma parameters; there is no separate scalar initial_value in this model
    optimizer = torch.optim.Adam([
        {"params": value_parameters, "lr": training_config.initial_value_lr},
        {"params": other_parameters, "lr": training_config.learning_rate},
    ])
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    report_every = max(training_config.epochs // 10, 1)
    was_training = model.training
    model.train()

    try:
        for epoch in range(1, training_config.epochs + 1):
            start_step = 0
            if sample_start_times and torch.rand((), device=device).item() >= initial_start_probability:
                start_step = int(torch.randint(cfg.num_time_steps, (), device=device).item())
            # The uniform branch includes zero; start_step is always an absolute fine-grid index.
            dI = math.sqrt(cfg.dt) * torch.randn(
                training_config.batch_size, cfg.num_time_steps - start_step,
                cfg.observation_dim, dtype=dtype, device=device,
            ) # [batch, remaining_steps, observation_dim], component variance dt
            result = model(dI, start_step=start_step, return_training=True)

            target_value = model.problem.terminal_cost(result["S_T"]) # [batch]
            target_gradient = model.geometry.project_gradient(
                model.problem.terminal_gradient(result["S_T"])
            ) # [batch, state_dim], posterior derivative projected onto the zero-sum tangent space
            terminal_value_loss = (result["R_T"] - target_value).square().mean()
            terminal_gradient_loss = (result["Z_T"] - target_gradient).square().mean() # mean over batch and state coordinates
            total_loss = (
                training_config.value_loss_weight * terminal_value_loss
                + training_config.gradient_loss_weight * terminal_gradient_loss
                + consistency_weight * result["consistency_loss"]
                + boundary_weight * result["boundary_loss"]
                + gradient_consistency_weight * result["gradient_consistency_loss"]
            ) # both sides of consistency terms remain differentiable; raw auxiliary cost is not in the loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()

            metrics = {
                "total_loss": total_loss,
                "terminal_value_loss": terminal_value_loss,
                "terminal_gradient_loss": terminal_gradient_loss,
                "consistency_loss": result["consistency_loss"],
                "boundary_loss": result["boundary_loss"],
                "gradient_consistency_loss": result["gradient_consistency_loss"],
                "mean_initial_value": result["initial_values"].mean(),
                "mean_auxiliary_cost": result["raw_auxiliary_cost"].mean(),
                "mean_gap": result["gap_integral"].mean(),
            }
            for name, metric in metrics.items():
                getattr(history, name).append(float(metric.detach().cpu()))
            history.start_step.append(start_step)

            if verbose and (epoch == 1 or epoch % report_every == 0):
                print(
                    f"epoch={epoch:4d}  start_step={start_step:3d}  "
                    f"total_loss={history.total_loss[-1]:.3e}  "
                    f"terminal_value={history.terminal_value_loss[-1]:.3e}  "
                    f"terminal_gradient={history.terminal_gradient_loss[-1]:.3e}  "
                    f"consistency={history.consistency_loss[-1]:.3e}  "
                    f"boundary={history.boundary_loss[-1]:.3e}  "
                    f"mean_start_value={history.mean_initial_value[-1]:.6f}"
                )
    finally:
        model.train(was_training) # restore the caller's train/eval mode, including after an error

    return history
