import torch
from torch import Tensor

from tree_expectations import N, R, S
from tree_expectations.matrix_tree_theorem import lap, dlap, matrix_tree_theorem, _dz_base


def _mu(w: 'Tensor[N, N]', B: 'Tensor[N, N]' = None) -> 'Tensor[N, N]':
    return w * _dz_base(w, B)


def zeroth_order(w: 'Tensor[N, N]', q: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the zeroth-order expectation of multiplicatively decomsosable
    function q.
    This algorithm is E_0 in the paper.
    This function has a runtime of O(N^3).
    """
    w_q = w * q
    return matrix_tree_theorem(w_q) / matrix_tree_theorem(w)


def first_order(w: 'Tensor[N, N]', r: 'Tensor[N, N, R]') -> 'Tensor[R]':
    """
    Compute the first-order expectation of additively decomsosable function r.
    This algorithm is E_1 in the paper.
    This function has a runtime of O(N^3 + N^2 R').
    """
    n = w.size(0)
    B = torch.inverse(lap(w)).t()
    mu = _mu(w, B)
    return (mu.unsqueeze(-1) * r).reshape(n*n, -1).sum(0)


def second_order(
        w: 'Tensor[N, N]',
        r: 'Tensor[N, N, R]',
        s: 'Tensor[N, N, S]'
) -> 'Tensor[R, S]':
    """
    Compute the second-order expectation of additively decomsosable
    functions r and s.
    This algorithm is E_2 in the paper.
    This function has a runtime of:
        O(N^3 (R' + S') + R S + N^2 min(R, N R') min(S, n S'))
    """
    n = w.size(0)
    rdim = r.size(-1)
    sdim = s.size(-1)
    B = torch.inverse(lap(w)).t()
    mu = _mu(w, B)
    e_r = (mu.unsqueeze(-1) * r).reshape(n*n, -1).sum(0)
    e_s = (mu.unsqueeze(-1) * s).reshape(n*n, -1).sum(0)
    rhat = torch.zeros((n, n, rdim)).double().requires_grad_(True)
    shat = torch.zeros((n, n, sdim)).double().requires_grad_(True)
    e_t = torch.ger(e_r, e_s)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for i_, j_, dL in dlap(i, j):
                    rhat[k, j_] += B[i_, k] * dL * w[i, j] * r[i, j]
                    shat[j_, k] += B[i_, k] * dL * w[i, j] * s[i, j]
    for i in range(n):
        for j in range(n):
            e_t += mu[i, j] * torch.ger(r[i, j], s[i, j]) - torch.ger(rhat[i, j], shat[i, j])
    return e_t


def covariance(
        w: 'Tensor[N, N]',
        r: 'Tensor[N, N, R]',
        s: 'Tensor[N, N, S]'
) -> 'Tensor[R, S]':
    """
    Compute the covariance between additively decomsosable functions r and s.
    This function has a runtime of:
        O(N^3 (R' + S') + N^2 min(R, N R') min(S, n S'))
    """
    n = w.size(0)
    rdim = r.size(-1)
    sdim = s.size(-1)
    B = torch.inverse(lap(w)).t()
    mu = _mu(w, B)
    rhat = torch.zeros((n, n, rdim)).double().requires_grad_(True)
    shat = torch.zeros((n, n, sdim)).double().requires_grad_(True)
    cov = torch.zeros((rdim, sdim)).double().requires_grad_(True)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for i_, j_, dL in dlap(i, j):
                    rhat[k, j_] += B[i_, k] * dL * w[i, j] * r[i, j]
                    shat[j_, k] += B[i_, k] * dL * w[i, j] * s[i, j]
    for i in range(n):
        for j in range(n):
            cov[:, :] += mu[i, j] * torch.ger(r[i, j], s[i, j]) - torch.ger(rhat[i, j], shat[i, j])
    return cov


def zeroth_order_grad(w: 'Tensor[N, N]', q: 'Tensor[N, N]') -> 'Tensor[N, N]':
    """
    Compute the gradient of a zeroth-order expectation.
    This assumes that q does not depend on w, if this is not the case,
    the gradient of q must be added to grad
    This relates to Proposition 4 in the paper.
    This function has a runtime of O(N^3)
    """
    wq = w * q
    B = torch.inverse(lap(w)).t()
    B_q = torch.inverse(lap(wq)).t()
    base = _dz_base(w, B)
    base_q = _dz_base(wq, B_q) * q
    grad = zeroth_order(w, q) * (base_q - base)
    return grad


def first_order_grad(w: 'Tensor[N, N]', r: 'Tensor[N, N, R]') -> 'Tensor[R, N, N]':
    """
    Compute the gradient of a first-order expectation.
    This assumes that r does not depend on w, if this is not the case,
    the gradient of r must be added to grad
    This relates to Theorem 3 in the paper.
    This function has a runtime of O(N^3 min(R, N R'))
    """
    n = w.size(0)
    rdim = r.size(-1)
    B = torch.inverse(lap(w)).t()
    base = _dz_base(w, B)
    rhat = torch.zeros((n, n, rdim)).double().requires_grad_(True)
    shat = torch.zeros((n, n, n, n)).double().requires_grad_(True)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for i_, j_, dL in dlap(i, j):
                    rhat[k, j_] += B[i_, k] * dL * w[i, j] * r[i, j]
                    shat[j_, k, i, j] += B[i_, k] * dL
    shat = shat.reshape(n, n, n*n)
    e_rs = torch.zeros((rdim, n, n)).double().requires_grad_(True)
    hats = torch.zeros((rdim, n*n)).double().requires_grad_(True)
    for i in range(n):
        for j in range(n):
            e_rs[:, i, j] += base[i, j] * r[i, j]
            hats[:, :] += torch.ger(rhat[i, j], shat[i, j])
    e_rs = e_rs.reshape(rdim, n*n)
    return e_rs - hats
