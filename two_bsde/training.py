"""Outer 2BSDE training loop."""
from dataclasses import dataclass, field
import math
import torch


@dataclass
class TrainingHistory:
    total_loss: list = field(default_factory=list)
    value_loss: list = field(default_factory=list)
    gradient_loss: list = field(default_factory=list)


def train(model, *, model_config, training_config, verbose=True):
    history = TrainingHistory()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config.learning_rate
    )

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    report_every = max(training_config.epochs // 10, 1)

    for epoch in range(1, training_config.epochs + 1):
        dI = math.sqrt(model_config.dt) * torch.randn(
            training_config.batch_size,
            model_config.num_time_steps,
            model_config.observation_dim,
            dtype=dtype,
            device=device,
        )

        R_T, Z_T, S_T, corr = model(dI, return_paths=False)

        target_value = model.problem.terminal_cost(S_T)
        target_gradient = model.geometry.project_gradient(
            model.problem.terminal_gradient(S_T)
        )

        value_loss = (R_T - target_value).square().mean()
        gradient_loss = (Z_T - target_gradient).square().mean()
        total_loss = (
            training_config.value_loss_weight * value_loss
            + training_config.gradient_loss_weight * gradient_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.gradient_clip
        )
        optimizer.step()

        history.total_loss.append(float(total_loss.detach().cpu()))
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
