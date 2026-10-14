"""Simplex-tangent parameterisations for gradients and Hessians."""
from curses import raw
from dataclasses import dataclass
import torch

# The posterior has num_states entries summing to one, so its tangent vectors sum to zero.
# Q denotes the basis below; q = num_states - 1 is the number of tangent coordinates.
# Shapes below use [batch, ...]; additional leading dimensions are also supported.


def make_simplex_tangent_basis(num_states, *, dtype, device):
    """
        output: [num_states, num_states-1], orthonormal columns spanning the zero-sum subspace
    """
    eye = torch.eye(num_states, dtype=dtype, device=device) # [num_states, num_states]

    # eye[:, :-1], shape [num_states, num_states-1]
    # eye[:, -1:], shape [num_states, 1], last coordinate vector
    spanning = eye[:, :-1] - eye[:, -1:].expand(-1, num_states-1) # columns e_i - e_last, each sums to zero
    basis, _ = torch.linalg.qr(spanning, mode="reduced") # orthonormalise the columns without changing their span
    return basis # Q [num_states, num_states-1], Q.T Q = I and each column sums to zero


def _upper_triangular_indices(dim): # include the diagonal, ordered row by row
    return [(i, j) for i in range(dim) for j in range(i, dim)] # dim*(dim+1)//2 pairs


def _raw_to_symmetric(raw, dim): # construct symmetric matrix from raw output of neural network
    """
        raw: [batch, dim*(dim+1)//2], independent upper-triangular entries
        output: [batch, dim, dim], symmetric matrix
    """
    indices = _upper_triangular_indices(dim)
    out = torch.zeros(
        *raw.shape[:-1], dim, dim, dtype=raw.dtype, device=raw.device
    ) # [batch, dim, dim]
    for k, (i, j) in enumerate(indices):
        out[..., i, j] = raw[..., k] # fill the upper triangle
        out[..., j, i] = raw[..., k] # mirror the entry; diagonal entries are assigned the same value twice
    return out # symmetry is enforced; positive semidefiniteness is not required for a Hessian


@dataclass(frozen=True)
class SimplexGeometry:
    observation_dim: int # m, number of observation coordinates
    basis: torch.Tensor # Q [num_states, tangent_dim], orthonormal basis of the posterior zero-sum subspace

    @property
    def num_states(self):
        return self.basis.shape[0] # number of posterior probabilities

    @property
    def tangent_dim(self):
        return self.basis.shape[1] # q = num_states-1 for the basis constructed above

    @property
    def state_dim(self):
        return self.observation_dim + self.num_states # full state S = (Y, p)

    @property
    def projector(self):
        return self.basis @ self.basis.T # P = Q Q.T [num_states, num_states], orthogonal projection onto zero-sum vectors

    def tangent_from_coordinates(self, raw_coordinates):
        """
            raw_coordinates: [batch, tangent_dim], arbitrary coordinates, with no sum constraint
            output: [batch, num_states], zero-sum posterior tangent vector
        """
        # self.basis, shape [num_states, tangent_dim]
        return torch.einsum("...a,ia->...i", raw_coordinates, self.basis) # Q times the coordinate vector

    def tangent_coordinates(self, tangent_vector):
        """
            tangent_vector: [batch, num_states], zero-sum vector in the tangent space
            output: [batch, tangent_dim], its coordinates in the orthonormal basis Q
        """
        return torch.einsum("...i,ia->...a", tangent_vector, self.basis) # Q.T times the tangent vector

    def project_p_vector(self, vector):
        """
            vector: [batch, num_states], arbitrary posterior-coordinate vector
            output: [batch, num_states], zero-sum projection, not a probability vector
        """
        return torch.einsum("ij,...j->...i", self.projector, vector) # P times vector, removes its component along the all-ones vector

    def vector_from_raw(self, raw): # convert reduced gradient or drift coordinates to the full state dimension
        """
            raw: [batch, observation_dim+tangent_dim], reduced coordinates of Z or A
            output: [batch, state_dim], observation part unchanged, posterior part zero-sum
        """
        m = self.observation_dim
        return torch.cat(
            (raw[..., :m], self.tangent_from_coordinates(raw[..., m:])),
            dim=-1,
        ) # concatenate observation coordinates with Q times the posterior coordinates

    def vector_to_raw(self, tangent_vector): # inverse of vector_from_raw when the posterior part is zero-sum
        """
            tangent_vector: [batch, state_dim], full vector Z or A, not the state S itself
            output: [batch, observation_dim+tangent_dim], reduced coordinates
        """
        m = self.observation_dim
        return torch.cat(
            (tangent_vector[..., :m], self.tangent_coordinates(tangent_vector[..., m:])),
            dim=-1,
        ) # observation part unchanged, posterior part multiplied by Q.T

    def gamma_from_raw(self, raw): # construct the full Gamma matrix from reduced symmetric coefficients
        """
            raw: [batch, gamma_parameter_dim], packed yy, C and H coefficients
            output: [batch, state_dim, state_dim], symmetric Gamma with tangent posterior blocks
        """
        m, q = self.observation_dim, self.tangent_dim
        yy_n = m * (m + 1) // 2 # independent entries of the observation-observation block
        yp_n = m * q # entries of the observation-tangent block C
        pp_n = q * (q + 1) // 2 # independent entries of the tangent-tangent block H
        expected = yy_n + yp_n + pp_n
        if raw.shape[-1] != expected:
            raise ValueError(f"Expected raw Gamma dim {expected}, got {raw.shape[-1]}")

        """
            The reduced symmetric matrix has blocks yy, C, C.T and H.
            In full posterior coordinates the blocks are yy, C Q.T, Q C.T and Q H Q.T.
        """
        o = 0 # offset into the packed coefficient vector
        yy = _raw_to_symmetric(raw[..., o:o+yy_n], m) # [batch, observation_dim, observation_dim]
        o += yy_n
        C = raw[..., o:o+yp_n].reshape(*raw.shape[:-1], m, q) # [batch, observation_dim, tangent_dim]
        o += yp_n
        H = _raw_to_symmetric(raw[..., o:o+pp_n], q) # [batch, tangent_dim, tangent_dim]

        # C,          shape [batch, observation_dim, tangent_dim]
        # self.basis, shape [num_states, tangent_dim]
        yp = torch.einsum("...ma,ia->...mi", C, self.basis) # C Q.T [batch, observation_dim, num_states], each row sums to zero
        pp = torch.einsum("ia,...ab,jb->...ij", self.basis, H, self.basis) # Q H Q.T [batch, num_states, num_states], rows and columns sum to zero

        gamma = torch.zeros(
            *raw.shape[:-1], self.state_dim, self.state_dim,
            dtype=raw.dtype, device=raw.device
        ) # [batch, state_dim, state_dim]
        gamma[..., :m, :m] = yy # observation-observation block
        gamma[..., :m, m:] = yp # observation-posterior block
        gamma[..., m:, :m] = yp.transpose(-1, -2) # posterior-observation block, transpose ensures symmetry
        gamma[..., m:, m:] = pp # posterior-posterior block
        return gamma

    def project_gradient(self, gradient):
        """
            gradient: [batch, state_dim], full gradient of a function of (Y, p)
            output: [batch, state_dim], observation part unchanged, posterior part projected
        """
        m = self.observation_dim
        return torch.cat(
            (gradient[..., :m], self.project_p_vector(gradient[..., m:])),
            dim=-1,
        ) # diag(I_m, P) times gradient, gives the zero-sum representative of the posterior derivative

    def project_hessian(self, hessian):
        """
            hessian: [batch, state_dim, state_dim], full matrix to symmetrise and project
            output: [batch, state_dim, state_dim], symmetric matrix with tangent posterior blocks
        """
        m = self.observation_dim
        g = 0.5 * (hessian + hessian.transpose(-1, -2)) # [batch, state_dim, state_dim], symmetric part
        P = self.projector # [num_states, num_states]
        yy = g[..., :m, :m] # [batch, observation_dim, observation_dim], already symmetric
        yp = torch.einsum("...mi,ij->...mj", g[..., :m, m:], P) # [batch, observation_dim, num_states], project each row
        pp = torch.einsum("ij,...jk,kl->...il", P, g[..., m:, m:], P) # [batch, num_states, num_states], project rows and columns
        out = torch.zeros_like(g)
        out[..., :m, :m] = yy
        out[..., :m, m:] = yp
        out[..., m:, :m] = yp.transpose(-1, -2)
        out[..., m:, m:] = pp
        return out # diag(I_m, P) @ g @ diag(I_m, P)

    def residuals(self, gradient, gamma, drift_gradient=None):
        """
            gradient: [batch, state_dim], Z; gamma: [batch, state_dim, state_dim], Gamma
            drift_gradient: [batch, state_dim], optional drift A of Z
            output: dictionary of maximum absolute constraint violations as Python floats
        """
        m = self.observation_dim
        # Each maximum covers all remaining coordinates and leading batch dimensions.
        # These are geometric constraint checks, expected to be near zero up to rounding.
        r = {
            "z_p_sum": float(gradient[..., m:].sum(-1).abs().max().detach().cpu()), # posterior components of Z sum to zero
            "gamma_symmetry": float(
                (gamma - gamma.transpose(-1, -2)).abs().max().detach().cpu()
            ), # largest entrywise difference from the transpose
            "gamma_yp_row_sum": float(
                gamma[..., :m, m:].sum(-1).abs().max().detach().cpu()
            ), # row sums of the observation-posterior block
            "gamma_pp_row_sum": float(
                gamma[..., m:, m:].sum(-1).abs().max().detach().cpu()
            ), # row sums of the posterior-posterior block
            "gamma_pp_col_sum": float(
                gamma[..., m:, m:].sum(-2).abs().max().detach().cpu()
            ), # column sums of the posterior-posterior block
        }
        if drift_gradient is not None:
            r["A_p_sum"] = float(
                drift_gradient[..., m:].sum(-1).abs().max().detach().cpu()
            ) # posterior components of A sum to zero
        return r
