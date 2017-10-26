# -*- coding: utf-8 -*-
"""Kernel functions."""

import six
import abc
import theano.tensor as T


@six.add_metaclass(abc.ABCMeta)
class Kernel():
    """Base Kernel."""

    @abc.abstractmethod
    def __call__(self, xi, xj):
        """Kernel function."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def hyperparameters(self):
        """Hyperparameters."""
        raise NotImplementedError()


class RBFKernel(Kernel):
    """Radial-Basis Function Kernel."""

    def __init__(self, length_scale, sigma_s, sigma_n):
        """Construct an RBFKernel.

        Args:
            length_scale (SharedVariable<dscalar>): Length scale.
            sigma_s (SharedVariable<dscalar>): Signal variance.
            sigma_n (SharedVariable<dscalar>): Noise variance.
        """
        self._sigma_s = sigma_s
        self._sigma_n = sigma_n
        self._length_scale = length_scale

        self._hyperparameters = [
            p for p in (length_scale, sigma_s, sigma_n)
            if isinstance(p, T.sharedvar.SharedVariable)
        ]

    def __call__(self, xi, xj):
        """Kernel function."""
        xi_squared = T.sum(xi**2, axis=1).reshape((-1, 1))
        xj_squared = T.sum(xj**2, axis=1).reshape((1, -1))
        dist_squared = xi_squared - 2 * T.dot(xi, xj.T) + xj_squared
        k = T.exp(-0.5 * (1 / self._length_scale) * dist_squared)
        return self._sigma_s**2 * k + self._sigma_n**2

    @property
    def hyperparameters(self):
        """Hyperparameters."""
        return self._hyperparameters
