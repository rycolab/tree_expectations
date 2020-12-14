import unittest
import torch

from tree_expectations.matrix_tree_theorem import matrix_tree_theorem
from tree_expectations.expectation import zeroth_order, zeroth_order_grad
from tree_expectations.expectation import first_order, first_order_grad
from tree_expectations.expectation import second_order, covariance
from tree_expectations.brute_force import bf_mtt, bf_zeroth, bf_first, bf_second
from tree_expectations.brute_force import bf_shannon_entropy, bf_kl
from tree_expectations.applications import shannon_entropy, entropy_grad
from tree_expectations.applications import kl_divergence, kl_grad
from tree_expectations.applications import ge_objective, ge_grad


class TreeExpectationTests(unittest.TestCase):
    def test_matrix_tree_theorem(self):
        """
        Test Matrix Tree Theorem
        """
        for n in range(3, 6):
            for _ in range(5):
                P = torch.exp(torch.randn(n, n)).double()
                P = P.clone().detach().requires_grad_(True)
                P_brute = P.clone().detach().requires_grad_(True)
                Z = matrix_tree_theorem(P)
                Z_brute = bf_mtt(P_brute)
                self.assertTrue(torch.allclose(Z, Z_brute))

    def test_zeroth(self):
        """
        Test random zeroth-order expectations
        """
        for n in range(3, 6):
            for _ in range(3):
                w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                q = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                w_brute = w.clone().detach().requires_grad_(True)
                q_brute = q.clone().detach().requires_grad_(True)
                e = zeroth_order(w, q)
                e_brute = bf_zeroth(w_brute, q_brute)
                self.assertTrue(torch.allclose(e, e_brute))

    def test_first(self):
        """
        Test random first-order expectations
        """
        for n in range(3, 6):
            for rdim in range(3, 6):
                for _ in range(3):
                    w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                    r = torch.exp(torch.randn(n, n, rdim)).double().requires_grad_(True)
                    w_brute = w.clone().detach().requires_grad_(True)
                    r_brute = r.clone().detach().requires_grad_(True)
                    e = first_order(w, r)
                    e_brute = bf_first(w_brute, r_brute)
                    self.assertTrue(torch.allclose(e, e_brute))

    def test_second(self):
        """
        Test random second-order expectations
        """
        for n in range(3, 6):
            for rdim in range(3, 6):
                for sdim in range(3, 6):
                    for _ in range(3):
                        w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                        r = torch.exp(torch.randn(n, n, rdim)).double().requires_grad_(True)
                        s = torch.exp(torch.randn(n, n, rdim)).double().requires_grad_(True)
                        w_brute = w.clone().detach().requires_grad_(True)
                        r_brute = r.clone().detach().requires_grad_(True)
                        s_brute = s.clone().detach().requires_grad_(True)
                        e = second_order(w, r, s)
                        e_brute = bf_second(w_brute, r_brute, s_brute)
                        self.assertTrue(torch.allclose(e, e_brute))

    def test_zeroth_grad(self):
        """
        Test random gradients of zeroth-order expectations
        """
        for n in range(3, 10):
            for _ in range(5):
                w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                q = torch.randn(n, n).double().requires_grad_(True)

                e = zeroth_order(w, q)
                true_grad = torch.autograd.grad(e, [w], retain_graph=True, create_graph=True)[0]

                grad = zeroth_order_grad(w, q)
                self.assertTrue(torch.allclose(grad, true_grad, rtol=1e-4))

    def test_first_grad_as_second(self):
        """
        Test random gradients of first-order expectations
        """
        for n in range(3, 10):
            for rdim in range(3, 10):
                w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                r = torch.exp(torch.randn(n, n, rdim)).double().requires_grad_(True)
                e = first_order(w, r)
                e_grad = torch.zeros((rdim, n, n)).double()
                for i in range(rdim):
                    e_grad[i] = torch.autograd.grad(e[i], [w], retain_graph=True, create_graph=True)[0]
                e_grad = e_grad.reshape(rdim, n*n)
                s = torch.zeros((n, n, n, n)).double().requires_grad_(True)
                for i in range(n):
                    for j in range(n):
                        s[i, j, i, j] = 1. / w[i, j]
                s = s.reshape(n, n, n*n)
                cov = covariance(w, r, s)
                grad = first_order_grad(w, r)
                self.assertTrue(torch.allclose(e_grad, cov))
                self.assertTrue(torch.allclose(e_grad, grad))

    def test_entropy(self):
        """
        Test value of the Shannon entropy
        """
        for n in range(3, 6):
            for _ in range(3):
                w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                ent = shannon_entropy(w)
                true_ent = bf_shannon_entropy(w)
                self.assertTrue(torch.allclose(true_ent, ent))

    def test_entropy_grad(self):
        """
        Test gradient of the Shannon entropy
        """
        for n in range(3, 10):
            for _ in range(3):
                w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                ent = shannon_entropy(w)
                true_grad = torch.autograd.grad(ent, [w], retain_graph=True, create_graph=True)[0]
                true_grad = true_grad.reshape(n * n)
                grad = entropy_grad(w)
                self.assertTrue(torch.allclose(true_grad, grad, rtol=1e-4))

    def test_kl(self):
        """
        Test value of the KL Divergence
        """
        for n in range(3, 6):
            for _ in range(3):
                w_p = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                w_q = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                kl = kl_divergence(w_p, w_q)
                true_kl = bf_kl(w_p, w_q)
                self.assertTrue(torch.allclose(true_kl, kl, rtol=1e-5))

    def test_kl_grad(self):
        """
        Test gradient of the KL Divergence
        """
        for n in range(3, 8):
            for _ in range(3):
                w_p = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                w_q = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                ent = kl_divergence(w_p, w_q)
                true_grad = torch.autograd.grad(ent, [w_p], retain_graph=True, create_graph=True)[0]
                true_grad = true_grad.reshape(n * n)
                grad = kl_grad(w_p, w_q)
                self.assertTrue(torch.allclose(true_grad, grad))

    def test_ge_grad(self):
        """
        Test gradient of the Generalized Expectation Criterion
        """
        for n in range(3, 10):
            for sdim in range(2, 6):
                for _ in range(3):
                    w = torch.exp(torch.randn(n, n)).double().requires_grad_(True)
                    s = torch.exp(torch.randn(n, n, sdim)).double().requires_grad_(True)
                    target = torch.randn(sdim).double().requires_grad_(True)
                    ge = ge_objective(w, s, target)
                    true_grad = torch.autograd.grad(ge, [w], retain_graph=True, create_graph=True)[0]
                    true_grad = true_grad.reshape(n * n)
                    grad = ge_grad(w, s, target)
                    self.assertTrue(torch.allclose(true_grad, grad))


if __name__ == '__main__':
    unittest.main()
