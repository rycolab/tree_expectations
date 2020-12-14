import torch
from torch import Tensor
from typing import List

from tree_expectations import F, N, R, S
from tree_expectations.utils import device


# Scores
def _prod_score(r: 'Tensor[N, N, R]', tree: 'Tensor[N]') -> 'Tensor[R]':
    n = r.size(0)
    return r[tree, torch.arange(n)].prod(0)


def _sum_score(r: 'Tensor[N, N, R]', tree: 'Tensor[N]') -> 'Tensor[R]':
    n = r.size(0)
    return r[tree, torch.arange(n)].sum(0)


def _enumerate_trees(A: 'Tensor[N, N]', root: int, root_weight: float) -> List:
    n = A.size(0)

    def enum_dst(weight, included, rest, excluded):
        if len(included) == n:
            return [(rest, weight)]
        dsts = []
        new_excluded = list(excluded)
        for i in included:
            for j in range(n):
                weight_ij = A[i, j]
                if j not in included and (i, j) not in excluded and weight_ij:
                    new_excluded += [(i, j)]
                    dsts += enum_dst(weight * weight_ij, included + [j],
                                     rest + [(i, j, weight_ij)], new_excluded)
        return dsts
    return enum_dst(root_weight, [root], [], [])


def all_multi_root_trees(w: 'Tensor[N, N]') -> List:
    """
    Compute all spanning trees of w that contain at least one root.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    n = w.size(0)
    rho = torch.diag(w)
    A = w * (torch.ones(1) - torch.eye(n)).to(device)
    new_A = torch.zeros((n + 1, n + 1))
    new_A[1:, 1:] = A
    new_A[0, 1:] = rho
    dsts = []
    unrooted_dsts = []
    for i in range(n):
        unrooted_dsts += _enumerate_trees(new_A, i, 1)
    for tree, weight in unrooted_dsts:
        t = - torch.ones(n)
        for i, j, _ in tree[1:]:
            if i == 0:
                t[j - 1] = j - 1
            else:
                t[j - 1] = i - 1
        dsts.append((t, weight))
    return dsts


def all_single_root_trees(w: 'Tensor[N, N]') -> List:
    """
    Compute all spanning trees of w that contain only one root.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    n = w.size(0)
    rho = torch.diag(w)
    A = w * (torch.ones(1) - torch.eye(n)).to(device)
    dsts = []
    for root, weight in enumerate(rho):
        if weight:
            rooted_dsts = _enumerate_trees(A, root, weight)
            for r_tree, weight in rooted_dsts:
                tree = - torch.ones(rho.size(0), dtype=torch.long)
                tree[root] = root
                for i, j, _ in r_tree:
                    tree[j] = i
                dsts += [(tree, weight)]
    return dsts


def bf_mtt(w: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the sum of costs over all spanning trees in w.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    Z = torch.tensor(0).double().to(device)
    for _, weight in all_single_root_trees(w):
        Z += weight
    return Z


def bf_zeroth(w: 'Tensor[N, N]', q: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the zeroth-order expectation over all spanning trees in w
    given a multiplicatively decomposable function q.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    e = torch.tensor(0).double().to(device)
    Z = torch.tensor(0).double().to(device)
    for tree, weight in all_single_root_trees(w):
        Z += weight
        e += weight * _prod_score(q, tree)
    return e / Z


def bf_first(w: 'Tensor[N, N]', r: 'Tensor[N, N, R]') -> 'Tensor[R]':
    """
    Compute the first-order expectation over all spanning trees in w
    given an additively decomposable function r.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    rdim = r.size(-1)
    e = torch.zeros(rdim).double().to(device)
    Z = torch.tensor(0).double().to(device)
    for tree, weight in all_single_root_trees(w):
        Z += weight
        e += weight * _sum_score(r, tree)
    return e / Z


def bf_second(
        w: 'Tensor[N, N]',
        r: 'Tensor[N, N, R]',
        s: 'Tensor[N, N, S]'
) -> 'Tensor[R, S]':
    """
    Compute the second-order expectation over all spanning trees in w
    given additively decomposable functions r and s.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    rdim = r.size(-1)
    sdim = s.size(-1)
    e = torch.zeros((rdim, sdim)).double().to(device)
    Z = bf_mtt(w)
    for tree, weight in all_single_root_trees(w):
        e += weight / Z * torch.ger(_sum_score(r, tree), _sum_score(s, tree))
    return e


# Renyi Entropy
def bf_renyi_entropy(w: 'Tensor[N, N]', alpha: float) -> 'Tensor[1]':
    """
    Compute the Renyi entropy of w.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    Z = torch.zeros(1).double().to(device)
    H = torch.zeros(1).double().to(device)
    for _, weight in all_single_root_trees(w):
        Z += weight
        H += torch.pow(weight, alpha)
    return (torch.log(H) - alpha * torch.log(Z)) / (1 - alpha)


# RISK
def bf_risk(w: 'Tensor[N, N]', r: 'Tensor[N, N, R]') -> 'Tensor[R]':
    """
    Compute the risk of w with respect to additively decomposable function r.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    Z = torch.zeros(1).double().to(device)
    risk = torch.zeros(1).double().to(device)
    for tree, weight in all_single_root_trees(w):
        Z += weight
        risk += weight * _sum_score(r, tree)
    return risk / Z


# Shannon Entropy
def bf_shannon_entropy(w: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the Shannon entropy of w.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    Z = torch.zeros(1).double().to(device)
    H = torch.zeros(1).double().to(device)
    for _, weight in all_single_root_trees(w):
        Z += weight
        H += weight * torch.log(weight)
    return torch.log(Z) - H / Z


# KL Divergence
def bf_kl(w_p: 'Tensor[N, N]', w_q: 'Tensor[N, N]') -> 'Tensor[1]':
    """
    Compute the KL divergence between the distributions of w and X.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    total = torch.zeros(1).double().to(device)
    Z_w = torch.zeros(1).double().to(device)
    Z_x = torch.zeros(1).double().to(device)
    for tree, weight in all_single_root_trees(w_p):
        weight_q = _prod_score(w_q, tree)
        Z_w += weight
        Z_x += weight_q
        total += weight * (torch.log(weight) - torch.log(weight_q))
    return torch.log(Z_x) - torch.log(Z_w) + total / Z_w


# Generalized Expectation
def bf_ge(w: 'Tensor[N, N]', f: 'Tensor[N, N, F]', target: 'Tensor[F]') -> 'Tensor[1]':
    """
    Compute the Generalized-Expected criterion of w with respect to additively decomposable
    function s and a target.
    Warning: this method is very inefficient.
    It should only be used on small examples, e.g., for testing purposes.
    """
    e = torch.zeros(f.size(-1))
    Z = torch.tensor(0).double().to(device)
    for tree, weight in all_single_root_trees(w):
        Z += weight
        e += weight * _sum_score(f, tree)
    residual = e - target
    return 0.5 * residual @ residual
