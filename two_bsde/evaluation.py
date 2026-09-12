"""Fresh-path evaluation."""
from dataclasses import dataclass
import math
import torch


@dataclass
class EvaluationResult:
    paths: dict
    diagnostics: dict


def evaluate(model, *, num_paths=1000):
    cfg = model.config
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    dI = math.sqrt(cfg.dt) * torch.randn(
        num_paths, cfg.num_time_steps, cfg.observation_dim,
        dtype=dtype, device=device
    )
    with torch.no_grad():
        paths = model(dI, return_paths=True)

    diagnostics = {
        "posterior_correction_rate":
            float(paths["posterior_correction_rate"].detach().cpu()),
        "mean_u_star":
            float(paths["u_star_path"].mean().detach().cpu()),
    }
    return EvaluationResult(paths=paths, diagnostics=diagnostics)
