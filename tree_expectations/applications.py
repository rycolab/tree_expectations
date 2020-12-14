import torch
from torch import Tensor

from tree_expectations import N, R, F
from tree_expectations.matrix_tree_theorem import matrix_tree_theorem, _dz_base
from tree_expectations.expectation import zeroth_order, first_order, first_order_grad
from tree_expectations.utils import device


# Renyi Entropy
def renyi(w: 'Tensor[N, N]', alpha: float):
    w_alpha = w.pow(1 - alpha)
    return zeroth_order(w, w_alpha)


# RISK - Section 5.1 of the paper
def risk(w: 'Tensor[N, N]', r: 'Tensor[N, N, R]') -> 'Tensor[R]':
    """
    Compute the risk of W with respect to additively decomposable function r.
    Function has a runtime of O(N^3+R'N^2)
    Note that r is a constant with respect to w
    """
    return first_order(w, r)


def risk_grad(w: 'Tensor[N, N]', r: 'Tensor[N, N, R]') -> 'Tensor[R, N^2]':
    """
    Compute the risk of w with respect to additively decomposable function r.
    Function has a runtime of O(N^3 min(R, N R') )
    Note that r is a constant with respect to w
    """
    return first_order_grad(w, r)


# Shannon Entropy - Section 5.2 of the paper
def shannon_entropy(w: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the Shannon entropy of w.
    Function has a runtime of O(N^3)
    """
    n = w.size(0)
    Z = matrix_tree_theorem(w)
    logw = w.clone()
    logw[logw == 0] = 1
    logw = torch.log(Z) / n - torch.log(logw)
    return first_order(w, logw.unsqueeze(-1))


def entropy_grad(w: 'Tensor[N, N]') -> 'Tensor[N^2]':
    """
    Compute the gradient of the Shannon entropy of w.
    Function has a runtime of O(N^3)
    """
    n = w.size(0)
    logw = w.clone()
    logw[logw == 0] = 1
    Z = matrix_tree_theorem(w)
    r = torch.log(Z) / n - torch.log(logw)
    dz = _dz_base(w) * Z
    mu = dz * w
    x = mu.sum() / (n * Z) - 1
    return first_order_grad(w, r.unsqueeze(-1)) + dz.reshape(n * n) * x


def smith_eisner_shannon_entropy(w: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the Shannon entropy of W using method described in:
    https://www.cs.jhu.edu/~jason/papers/smith+eisner.emnlp07.pdf
    Function has a runtime of O(N^3)
    """
    n = w.size(0)
    h = torch.tensor(0).double().to(device)
    log_w = w.clone()
    log_w[log_w == 0] = 1.
    log_w = torch.log(log_w)
    Z = matrix_tree_theorem(w)
    for i in range(n):
        w_mod = torch.ones((n, n)).double().to(device)
        w_mod[:, i] = log_w[:, i]
        h += matrix_tree_theorem(w * w_mod)
    return torch.log(Z) - h / Z


# KL Divergence - Section 5.3 of the paper
def kl_divergence(w_p: 'Tensor[N, N]', w_q: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the KL divergence of w_p and w_q.
    Function has a runtime of O(N^3)
    """
    n = w_p.size(0)
    log_w_p = w_p.clone()
    log_w_q = w_q.clone()
    log_w_p[log_w_p == 0] = 1
    log_w_q[log_w_q == 0] = 1
    Z_p = matrix_tree_theorem(w_p)
    Z_q = matrix_tree_theorem(w_q)
    r = torch.log(log_w_p) - torch.log(log_w_q) + \
        (torch.log(Z_q) - torch.log(Z_p)) / n
    return first_order(w_p, r.unsqueeze(-1))


def kl_grad(w_p: 'Tensor[N, N]', w_q: 'Tensor[N, N]') -> 'Tensor[N^2]':
    """
    Compute the gradient of the  KL divergence of w_p and w_q.
    Function has a runtime of O(N^3)
    """
    n = w_p.size(0)
    log_w_p = w_p.clone()
    log_w_q = w_q.clone()
    log_w_p[log_w_p == 0] = 1
    log_w_q[log_w_q == 0] = 1
    Z_p = matrix_tree_theorem(w_p)
    Z_q = matrix_tree_theorem(w_q)
    r = torch.log(log_w_p) - torch.log(log_w_q) + \
        (torch.log(Z_q) - torch.log(Z_p)) / n
    dz = _dz_base(w_p) * Z_p
    x = torch.ones(1).double() - (dz * w_p).sum() / (n * Z_p)
    return first_order_grad(w_p, r.unsqueeze(-1)) + dz.reshape(n * n) * x


# Generalized Expectation Criterion - Section 5.4 of the paper
def ge_objective(
        w: 'Tensor[N, N]',
        f: 'Tensor[N, N, F]',
        target: 'Tensor[F]'
) -> 'Tensor[1]':
    """
    Compute the Generalized-Expected criterion of w with respect to
    additively decomposable function s and a target.
    Function has a runtime of O(N^3 + N^2 F')
    """
    e_s = first_order(w, f)
    distance = e_s - target
    return 0.5 * distance @ distance


def ge_grad(
        w: 'Tensor[N, N]',
        f: 'Tensor[N, N, F]',
        target: 'Tensor[F]'
) -> 'Tensor[N^2]':
    """
    Compute the Generalized-Expected criterion of w with respect to
    additively decomposable function s and a target.
    Function has a runtime of O(N^3 + N^2 F)
    """
    first = first_order(w, f)
    residual = first - target
    f = torch.einsum("ijs,s->ij", f, residual)
    return first_order_grad(w, f).unsqueeze(0)


