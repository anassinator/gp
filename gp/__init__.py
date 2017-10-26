# -*- coding: utf-8 -*-
"""Gaussian Process implementation for Theano."""

__version__ = "0.1.0"

from .kernel import Kernel, RBFKernel
from .gaussian_process import GaussianProcess

__all__ = ["GaussianProcess", "Kernel", "RBFKernel"]
