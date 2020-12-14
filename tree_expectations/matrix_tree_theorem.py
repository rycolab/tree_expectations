import torch
from torch import Tensor

from tree_expectations import N
from tree_expectations.utils import device


def laplacian(A: 'Tensor[N, N]', rho: 'Tensor[N]') -> 'Tensor[N, N]':
    """
    Compute the root-weighted Laplacian.
    Function has a runtime of O(N^2).
    """
    L = -A + torch.diag_embed(A.sum(dim=0)).to(device)
    L[0] = rho
    return L


def lap(w: 'Tensor[N, N]') -> 'Tensor[N, N]':
    """
    Compute the root-weighted Laplacian.
    Function has a runtime of O(N^2).
    """
    rho = torch.diag(w)
    A = w * (torch.tensor(1).double().to(device) - torch.eye(w.size(0)).double().to(device))
    return laplacian(A, rho)


def dlap(k, l):
    """
    Index over sparsity in the Jacobian of the Laplacian

    Given (k,l), return (i,j) such that for a

      dL[i,j] / ∂A[k,l] ≠ 0

    Done in O(1), see Proposition of paper (see README)
    """
    if k == l:
        return [(0, k, 1.)]
    out = []
    if l != 0:
        out.append((l, l, 1.))
    if k != 0:
        out.append((k, l, -1.))
    return out


def adj(A: 'Tensor[N, N]') -> 'Tensor[N, N]':
    """
    Compute the adjugate of a matrix A.
    The adjugate can be used for calculating the derivative of a determinants.
    Function has a runtime of O(N^3).
    """
    Ad = torch.slogdet(A)
    Ad = Ad[0] * torch.exp(Ad[1])
    return Ad * torch.inverse(A).t()


def matrix_tree_theorem(w: 'Tensor[N, N]', use_log: bool = False) -> 'Tensor[1]':
    """
    Compute the sum over all spanning trees in W using the Matrix--Tree Theorem.
    Function has a runtime of O(N^3).
    This relates to Section 2, Proposition 1 of paper (see README)
    """
    r = torch.diag(w)
    A = w * (torch.tensor(1).double().to(device) - torch.eye(w.size(0)).double().to(device))
    sign, logZ = torch.slogdet(laplacian(A, r))
    return logZ if use_log else sign * torch.exp(logZ)


def _dz_base(w: 'Tensor[N, N]', B: 'Tensor[N, N]' = None):
    """
    Evaluate the base of the derivative of the Matrix--Tree Theorem
    using a possibly cached transpoed inverse Laplacian matrix.
    To get the full derivative, we must multiply by Z.
    To get mu, we must multiply by w.
    This function runs in O(N^2) if B is provided and O(N^3) otherwise.
    """
    n = w.size(0)
    if B is None:
        B = torch.inverse(lap(w)).t()
    base = torch.zeros((n, n)).double().requires_grad_(True)
    for i in range(n):
        for j in range(n):
            for i_, j_, dL in dlap(i, j):
                base[i, j] += B[i_, j_] * dL
    return base
