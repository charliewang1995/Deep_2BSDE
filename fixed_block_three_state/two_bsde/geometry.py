"""Simplex-tangent parameterisations for gradients and Hessians."""
from dataclasses import dataclass
import torch


def make_simplex_tangent_basis(num_states, *, dtype, device):
    eye = torch.eye(num_states, dtype=dtype, device=device)
    spanning = eye[:, :-1] - eye[:, -1:].expand(-1, num_states - 1)
    basis, _ = torch.linalg.qr(spanning, mode="reduced")
    return basis


def _upper_triangular_indices(dim):
    return [(i, j) for i in range(dim) for j in range(i, dim)]


def _raw_to_symmetric(raw, dim):
    indices = _upper_triangular_indices(dim)
    out = torch.zeros(
        *raw.shape[:-1], dim, dim, dtype=raw.dtype, device=raw.device
    )
    for k, (i, j) in enumerate(indices):
        out[..., i, j] = raw[..., k]
        out[..., j, i] = raw[..., k]
    return out


@dataclass(frozen=True)
class SimplexGeometry:
    observation_dim: int
    basis: torch.Tensor

    @property
    def num_states(self):
        return self.basis.shape[0]

    @property
    def tangent_dim(self):
        return self.basis.shape[1]

    @property
    def state_dim(self):
        return self.observation_dim + self.num_states

    @property
    def projector(self):
        return self.basis @ self.basis.T

    def tangent_from_coordinates(self, coordinates):
        return torch.einsum("...a,ia->...i", coordinates, self.basis)

    def tangent_coordinates(self, ambient):
        return torch.einsum("...i,ia->...a", ambient, self.basis)

    def project_p_vector(self, vector):
        return torch.einsum("ij,...j->...i", self.projector, vector)

    def vector_from_raw(self, raw):
        m = self.observation_dim
        return torch.cat(
            (raw[..., :m], self.tangent_from_coordinates(raw[..., m:])),
            dim=-1,
        )

    def vector_to_raw(self, ambient):
        m = self.observation_dim
        return torch.cat(
            (ambient[..., :m], self.tangent_coordinates(ambient[..., m:])),
            dim=-1,
        )

    def gamma_from_raw(self, raw):
        m, q = self.observation_dim, self.tangent_dim
        yy_n = m * (m + 1) // 2
        yp_n = m * q
        pp_n = q * (q + 1) // 2
        expected = yy_n + yp_n + pp_n
        if raw.shape[-1] != expected:
            raise ValueError(f"Expected raw Gamma dim {expected}, got {raw.shape[-1]}")

        o = 0
        yy = _raw_to_symmetric(raw[..., o:o+yy_n], m)
        o += yy_n
        C = raw[..., o:o+yp_n].reshape(*raw.shape[:-1], m, q)
        o += yp_n
        H = _raw_to_symmetric(raw[..., o:o+pp_n], q)

        yp = torch.einsum("...ma,ia->...mi", C, self.basis)
        pp = torch.einsum("ia,...ab,jb->...ij", self.basis, H, self.basis)

        gamma = torch.zeros(
            *raw.shape[:-1], self.state_dim, self.state_dim,
            dtype=raw.dtype, device=raw.device
        )
        gamma[..., :m, :m] = yy
        gamma[..., :m, m:] = yp
        gamma[..., m:, :m] = yp.transpose(-1, -2)
        gamma[..., m:, m:] = pp
        return gamma

    def project_gradient(self, gradient):
        m = self.observation_dim
        return torch.cat(
            (gradient[..., :m], self.project_p_vector(gradient[..., m:])),
            dim=-1,
        )

    def project_hessian(self, hessian):
        m = self.observation_dim
        g = 0.5 * (hessian + hessian.transpose(-1, -2))
        P = self.projector
        yy = g[..., :m, :m]
        yp = torch.einsum("...mi,ij->...mj", g[..., :m, m:], P)
        pp = torch.einsum("ij,...jk,kl->...il", P, g[..., m:, m:], P)
        out = torch.zeros_like(g)
        out[..., :m, :m] = yy
        out[..., :m, m:] = yp
        out[..., m:, :m] = yp.transpose(-1, -2)
        out[..., m:, m:] = pp
        return out

    def residuals(self, gradient, gamma, drift_gradient=None):
        m = self.observation_dim
        r = {
            "z_p_sum": float(gradient[..., m:].sum(-1).abs().max().detach().cpu()),
            "gamma_symmetry": float(
                (gamma - gamma.transpose(-1, -2)).abs().max().detach().cpu()
            ),
            "gamma_yp_row_sum": float(
                gamma[..., :m, m:].sum(-1).abs().max().detach().cpu()
            ),
            "gamma_pp_row_sum": float(
                gamma[..., m:, m:].sum(-1).abs().max().detach().cpu()
            ),
            "gamma_pp_col_sum": float(
                gamma[..., m:, m:].sum(-2).abs().max().detach().cpu()
            ),
        }
        if drift_gradient is not None:
            r["A_p_sum"] = float(
                drift_gradient[..., m:].sum(-1).abs().max().detach().cpu()
            )
        return r
