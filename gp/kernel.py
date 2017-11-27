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

    @property
    @abc.abstractmethod
    def bounds(self):
        """List of minimum and maximum bounds for each hyperparameter."""
        raise NotImplementedError()


class RBFKernel(Kernel):

    """Radial-Basis Function Kernel."""

    def __init__(self,
                 length_scale,
                 sigma_s,
                 sigma_n,
                 bounds=[(1e-5, 1e6), (1e-5, 1.0), (1e-5, 1.0)]):
        """Construct an RBFKernel.

        Args:
            length_scale (SharedVariable<dscalar> or SharedVariable<dvector>):
                Length scale.
            sigma_s (SharedVariable<dscalar>): Signal standard deviation.
            sigma_n (SharedVariable<dscalar>): Noise standard deviation.
            bounds (list<tuple<float, float>>): Minimum and maximum bounds for
                each hyperparameter.
        """
        self._hyperparameters = [
            p for p in (length_scale, sigma_s, sigma_n)
            if isinstance(p, T.sharedvar.SharedVariable)
        ]

        self._sigma_s = sigma_s
        self._sigma_n = sigma_n
        self._length_scale = length_scale

        self._bounds = bounds

    def __call__(self, xi, xj):
        """Kernel function."""
        M = T.eye(xi.shape[1]) * self._length_scale**-2
        dist = mahalanobis(xi, xj, M)
        return self._sigma_s**2 * T.exp(-0.5 * dist) + self._sigma_n**2

    @property
    def hyperparameters(self):
        """Hyperparameters."""
        return self._hyperparameters

    @property
    def bounds(self):
        """List of minimum and maximum bounds for each hyperparameter."""
        return self._bounds


def mahalanobis(xi, xj, VI=None):
    """Computes the pair-wise squared mahalanobis distance matrix as:

        (xi - xj)^T V^-1 (xi - xj)

    Args:
        xi: xi input matrix.
        xj: xj input matrix.
        VI: The inverse of the covariance matrix, default: identity matrix.

    Returns:
        Weighted matrix of all pair-wise distances.
    """
    if VI is None:
        D = T.sum(T.square(xi), axis=1).reshape((-1, 1)) \
          + T.sum(T.square(xj), axis=1).reshape((1, -1)) \
          - 2 * xi.dot(xj.T)
    else:
        xi_VI = xi.dot(VI)
        xj_VI = xj.dot(VI)
        D = T.sum(xi_VI * xi, axis=1).reshape((-1, 1)) \
          + T.sum(xj_VI * xj, axis=1).reshape((1, -1)) \
          - 2 * xi_VI.dot(xj.T)
    return D
