"""Gaussian Process implementation."""

__version__ = "0.1.0"

import kernels
from .models import GaussianProcess, MultiGaussianProcess

__all__ = ["GaussianProcess", "MultiGaussianProcess", "kernels"]
