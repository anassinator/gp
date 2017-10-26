# -*- coding: utf-8 -*-
"""Gaussian Process."""

import theano
import theano.tensor as T
import theano.sandbox.linalg as sT


class GaussianProcess(object):
    """Gaussian Process regressor."""

    def __init__(self, kernel, sigma_n):
        """Constructs a GaussianProcess.

        Args:
            kernel (Kernel): Kernel to use.
            sigma_n (SharedVariable<dscalar>): Noise variance.
        """
        self._kernel = kernel
        self._sigma_n = sigma_n

        self._hyperparameters = [
            p for p in kernel.hyperparameters
            if p != sigma_n
        ]
        self._hyperparameters.append(sigma_n)

        self.X = None
        self.Y = None
        self._x = T.dmatrix("x")

    @property
    def hyperparameters(self):
        """Hyperparameters."""
        return self._hyperparameters

    def fit(self, X, Y):
        """Fits the gaussian process to the provided data.

        Args:
            X (SharedVariable<dmatrix>): Input data.
            Y (SharedVariable<dvector>): Output data.
        """
        self.X = X
        self.Y = Y
        x = self._x

        # Kernel functions.
        K_ss = self._kernel(x, x)
        K_s = self._kernel(x, X)
        K = self._kernel(X, X).eval() + self._sigma_n * T.eye(X.shape[0])

        # Mean and variance functions.
        K_inv = sT.matrix_inverse(K)
        mu = T.dot(K_s, T.dot(K_inv, Y))
        var = K_ss - T.dot(K_s, T.dot(K_inv, K_s.T))

        # Compute the standard deviation.
        L = sT.cholesky(K)
        L_k = sT.solve(L, K_s.T)
        std = T.sqrt(T.diag(K_ss) - T.sum(L_k**2, axis=0)).reshape((-1, 1))

        self._mu = mu
        self._var = var
        self._std = std

        # TODO: Find a better way to do this and verify scaling is correct.
        self._mu_grad = T.jacobian(mu.flatten(), x).sum(axis=0)
        self._var_grad = T.jacobian(var.flatten(), x).sum(axis=0)
        self._std_grad = T.jacobian(std.flatten(), x).sum(axis=0)

    @property
    def mean(self):
        """Expectation tensor variable."""
        return self._mu

    @property
    def variance(self):
        """Variance tensor variable."""
        return self._var

    @property
    def standard_deviation(self):
        """Standard deviation tensor variable."""
        return self._std

    def compile(self):
        """Compiles the gaussian process's tensors into proper functions."""
        inputs = [self._x]
        self._mu_f = theano.function(inputs, self._mu, name="mu")
        self._var_f = theano.function(inputs, self._var, name="var")
        self._std_f = theano.function(inputs, self._std, name="std")

        self._mu_grad_f = theano.function(
            inputs, self._mu_grad, name="mu_grad")
        self._var_grad_f = theano.function(
            inputs, self._var_grad, name="var_grad")
        self._std_grad_f = theano.function(
            inputs, self._std_grad, name="std_grad")

    def compute_mean(self, x):
        """Computes the expectation of x.

        Args:
            x (dvector): Inputs.

        Returns:
            Expectation of each input.
        """
        return self._mu_f(x)

    def compute_variance(self, x):
        """Computes the variance of x.

        Args:
            x (dvector): Inputs.

        Returns:
            Variance of each input.
        """
        return self._var_f(x)

    def compute_standard_deviation(self, x):
        """Computes the standard deviation of x.

        Args:
            x (dvector): Inputs.

        Returns:
            Standard deviation of each input.
        """
        return self._std_f(x)

    def compute_mean_grad(self, x):
        """Computes d/dx of the expectation of x.

        Args:
            x (dvector): Inputs.

        Returns:
            d/dx of the expectation of each input.
        """
        return self._mu_grad_f(x)

    def compute_variance_grad(self, x):
        """Computes d/dx of the variance of x.

        Args:
            x (dvector): Inputs.

        Returns:
            d/dx of the variance of each input.
        """
        return self._var_grad_f(x)

    def compute_standard_deviation_grad(self, x):
        """Computes d/dx of the standard deviation of x.

        Args:
            x (dvector): Inputs.

        Returns:
            d/dx of the standard deviation of each input.
        """
        return self._std_grad_f(x)
