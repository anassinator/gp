# -*- coding: utf-8 -*-
"""Gaussian Process."""

import theano
import numpy as np
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
            p for p in kernel.hyperparameters if p != sigma_n
        ]
        self._hyperparameters.append(sigma_n)

        self.X = T.dmatrix("X")
        self.Y = T.dmatrix("Y")
        self.x = T.dmatrix("x")
        self.X_train = None
        self.Y_train = None

        # Defaults.
        self._mu_f = self.__need_to_compile
        self._var_f = self.__need_to_compile
        self._std_f = self.__need_to_compile

        self.__setup()

    def __setup(self):
        """Sets up the gaussian process's tensor variables."""
        X = self.X
        Y = self.Y
        x = self.x

        # Kernel functions.
        K_ss = self._kernel(x, x)
        K_s = self._kernel(x, X)
        K = self._kernel(X, X) + self._sigma_n * T.eye(X.shape[0])

        # Mean and variance functions.
        K_inv = sT.matrix_inverse(K)
        mu = T.dot(K_s, T.dot(K_inv, Y))
        var = K_ss - T.dot(K_s, T.dot(K_inv, K_s.T))

        # Compute the standard deviation.
        L = sT.cholesky(K)
        L_k = T.slinalg.solve_lower_triangular(L, K_s.T)
        std = T.sqrt(T.diag(K_ss) - T.sum(L_k**2, axis=0)).reshape((-1, 1))

        # Compute the log likelihood.
        alpha = T.slinalg.solve_lower_triangular(L, Y)
        log_likelihood_dims = -0.5 * T.dot(Y.T, alpha).sum(axis=0)
        log_likelihood_dims -= T.log(T.diag(L)).sum()
        log_likelihood_dims -= L.shape[0] / 2 * T.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        self._mu = mu
        self._var = var
        self._std = std
        self._log_likelihood = log_likelihood

        # TODO: Find a better way to do this and verify scaling is correct.
        self._mu_grad = T.jacobian(mu.flatten(), x).sum(axis=0)
        self._var_grad = T.jacobian(var.flatten(), x).sum(axis=0)
        self._std_grad = T.jacobian(std.flatten(), x).sum(axis=0)

    def __need_to_compile(self, *args, **kwargs):
        """Helper function to verify compilation."""
        raise Exception("You must .compile() before calling this")

    @property
    def hyperparameters(self):
        """Hyperparameters."""
        return self._hyperparameters

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

    @property
    def log_likelihood(self):
        """Log likelihood tensor variable."""
        return self._log_likelihood

    def fit(self, X, Y):
        """Fits the model.

        Args:
            X (SharedVariable<dmatrix>): Input data.
            Y (SharedVariable<dvector>): Output data.
        """
        self.X_train = X
        self.Y_train = Y

        # TODO: Optimize hyperparameters.

    def compile(self):
        """Compiles the gaussian process's tensors into proper functions."""
        if None in (self.X_train, self.Y_train):
            raise Exception("You must .fit() before compiling")

        inputs = [self.x]
        givens = {
            self.X: self.X_train,
            self.Y: self.Y_train,
        }

        self._mu_f = theano.function(
            inputs,
            self._mu,
            name="mu",
            givens=givens,
            on_unused_input="ignore")
        self._var_f = theano.function(
            inputs,
            self._var,
            name="var",
            givens=givens,
            on_unused_input="ignore")
        self._std_f = theano.function(
            inputs,
            self._std,
            name="std",
            givens=givens,
            on_unused_input="ignore")

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

    def compute_log_likelihood(self, x):
        """Computes the log likelihood of x.

        Args:
            x (dvector): Inputs.

        Returns:
            Log likelihood of each input.
        """
        return self._log_likelihood_f(x)
