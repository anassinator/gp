# -*- coding: utf-8 -*-
"""Gaussian Process."""

import theano
import numpy as np
import theano.tensor as T
import theano.sandbox.linalg as sT

from .kernel import RBFKernel
from .optimize import optimize


class GaussianProcess(object):

    """
    Gaussian Process regressor.

    This is meant for multi-input, single-output functions.
    It still works for multi-output functions, but will share the same
    hyperparameters for each. If that is not wanted, use `MultiGaussianProcess`
    instead.
    """

    def __init__(self,
                 kernel=None,
                 sigma_n=None,
                 xdim=None,
                 bounds=(1e-5, 1.0),
                 normalize_y=True):
        """Constructs a GaussianProcess.

        Args:
            kernel (Kernel): Kernel to use, default: RBFKernel.
            sigma_n (SharedVariable<dscalar>): Noise standard deviation,
                default: random.
            xdim (int): Input dimension for length scale initialization if a
                kernel is not specified, default: use scalar length scale.
            bounds (tuple<float, float>): Minimum and maximum bounds for
                the sigma_n hyperparameter.
            normalize_y (bool): Normalize the Y values to have 0 mean and unit
                variance.
        """
        if sigma_n is None:
            sigma_n = theano.shared(
                np.random.random() / 10, name="sigma_n", borrow=True)

        if kernel is None:
            sigma_s = theano.shared(
                np.random.random(), name="sigma_s", borrow=True)
            length_scale = theano.shared(
                np.random.random(xdim), name="length_scale", borrow=True)
            kernel = RBFKernel(length_scale, sigma_s, sigma_n)

        self._kernel = kernel
        self._sigma_n = sigma_n

        self._hyperparameters = [
            p for p in kernel.hyperparameters if p != sigma_n
        ]
        self._hyperparameters.append(sigma_n)

        self._bounds = [
            kernel.bounds[i] for i, p in enumerate(kernel.hyperparameters)
            if p != sigma_n
        ]
        self._bounds.append(bounds)

        self.X = T.dmatrix("X")
        self.Y = T.dmatrix("Y")
        self.x = T.dmatrix("x")
        self.X_train = None
        self.Y_train = None

        # Defaults.
        self._mu_f = _need_to_compile
        self._var_f = _need_to_compile
        self._std_f = _need_to_compile

        self._normalize_y = normalize_y

        self._build_graph()

    def _build_graph(self):
        """Sets up the gaussian process's tensor variables."""
        X = self.X
        Y = self.Y
        x = self.x

        if self._normalize_y:
            Y_mean = T.mean(Y, axis=0)
            Y_variance = T.std(Y, axis=0)
            Y = (Y - Y_mean) / Y_variance

        # Kernel functions.
        K_ss = self._kernel(x, x)
        K_s = self._kernel(x, X)
        K = self._kernel(X, X) + self._sigma_n**2 * T.eye(X.shape[0])

        # Guarantee positive definite.
        K = 0.5 * (K + K.T)

        # Mean and variance functions.
        K_inv = sT.matrix_inverse(K)
        mu = T.dot(K_s, T.dot(K_inv, self.Y))  # Non-normalized Y for scale.
        var = K_ss - T.dot(K_s, T.dot(K_inv, K_s.T))

        # Compute the standard deviation.
        L = sT.cholesky(K)
        L_k = T.slinalg.solve_lower_triangular(L, K_s.T)
        std = T.sqrt(T.diag(K_ss) - T.sum(L_k**2, axis=0)).reshape((-1, 1))

        # Compute the log likelihood.
        log_likelihood_dims = -0.5 * T.dot(Y.T, T.dot(K_inv, Y)).sum(axis=0)
        log_likelihood_dims -= T.log(T.diag(L)).sum()
        log_likelihood_dims -= L.shape[0] / 2 * T.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        self._mu = mu
        self._var = var
        self._std = std
        self._log_likelihood = log_likelihood

    @property
    def hyperparameters(self):
        """Hyperparameters."""
        return self._hyperparameters

    @property
    def bounds(self):
        """List of minimum and maximum bounds for each hyperparameter."""
        return self._bounds

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

    def fit(self, X, Y, skip_optimization=False, *args, **kwargs):
        """Fits the model.

        Args:
            X (SharedVariable<dmatrix>): Input data.
            Y (SharedVariable<dvector>): Output data.
            skip_optimization (bool): Optimize the hyperparameters.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to `optimize()`.
        """
        self.X_train = X
        self.Y_train = Y

        if skip_optimization:
            return

        return optimize(self, *args, **kwargs)

    def compile(self):
        """Compiles the gaussian process's tensors into proper functions."""
        if self.X_train is None or self.Y_train is None:
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


class MultiGaussianProcess(GaussianProcess):

    """
    Layer of abstraction to estimate vector-valued functions with a separate
    Gaussian Process for each output dimension.

    This learns separate hyperparameters for each output dimension instead of
    a single set for all dimensions.
    """

    def __init__(self, ydim, kernels=None, *args, **kwargs):
        """Constructs a MultiGaussianProcess.

        Args: 
            ydim (int): Output dimension for number of Gaussian Processes to
                use.
            kernels (list<Kernel>): List of kernels to use for each dimension,
                default: RBFKernel for each.
            *args, **kwargs: Additional positional and key-word arguments to
                the Gaussian Process constructor.
        """
        if kernels is None:
            self._processes = [
                GaussianProcess(*args, **kwargs) for i in range(ydim)
            ]
        else:
            self._processes = [
                GaussianProcess(kernels[i], *args, **kwargs)
                for i in range(ydim)
            ]

        self.X = T.dmatrix("X")
        self.Y = T.dmatrix("Y")
        self.x = T.dmatrix("x")
        self.X_train = None
        self.Y_train = None

        self._hyperparameters = [
            p for gp in self._processes for p in gp.hyperparameters
        ]

        self._bounds = [
            gp.bounds[i]
            for gp in self._processes for i, p in enumerate(gp.hyperparameters)
        ]

        # Defaults.
        self._mu_f = _need_to_compile
        self._var_f = _need_to_compile
        self._std_f = _need_to_compile

        self._build_graph()

    def _build_graph(self):
        """Sets up the gaussian process's tensor variables."""
        # Rebuild the child gaussian processes' graphs with overrided, shared
        # variables.
        for gp in self._processes:
            gp.X = self.X
            gp.x = self.x
            gp._build_graph()

        # Concatenate the chidren's tensors for the parent.
        self._mu = T.concatenate([gp._mu for gp in self._processes], axis=1)
        self._std = T.concatenate([gp._std for gp in self._processes], axis=1)
        self._var = T.stack([gp._var for gp in self._processes], axis=2)
        self._log_likelihood = T.sum(
            [gp._log_likelihood for gp in self._processes])

    def fit(self, X, Y, *args, **kwargs):
        """Fits the model.

        Args:
            X (SharedVariable<dmatrix>): Input data.
            Y (SharedVariable<dvector>): Output data.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to each gaussian process's fit() function.
        """
        self.X_train = X
        self.Y_train = Y

        # Fitting them individually is faster than fitting them together as
        # they are completely independent of each other.
        for i, gp in enumerate(self._processes):
            gp.fit(X, Y[:, i].reshape((-1, 1)), *args, **kwargs)

    def compile(self):
        """Compiles the gaussian process's tensors into proper functions."""
        if self.X_train is None or self.Y_train is None:
            raise Exception("You must .fit() before compiling")

        inputs = [self.x]
        givens = {
            gp.Y: self.Y_train[:, i].reshape((-1, 1))
            for i, gp in enumerate(self._processes)
        }
        givens[self.X] = self.X_train

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


def _need_to_compile(*args, **kwargs):
    """Helper function to verify compilation."""
    raise Exception("You must .compile() before calling this")
