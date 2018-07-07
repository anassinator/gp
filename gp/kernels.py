import torch


class Kernel(torch.nn.Module):

    """Base kernel."""

    def __add__(self, other):
        """Sums two kernels together.

        Args:
            other (Kernel): Other kernel.

        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.add)

    def __mul__(self, other):
        """Multiplies two kernels together.

        Args:
            other (Kernel): Other kernel.

        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.mul)

    def __sub__(self, other):
        """Subtracts two kernels from each other.

        Args:
            other (Kernel): Other kernel.

        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.sub)

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        raise NotImplementedError


class AggregateKernel(Kernel):

    """An aggregate kernel."""

    def __init__(self, first, second, op):
        """Constructs an AggregateKernel.

        Args:
            first (Kernel): First kernel.
            second (Kernel): Second kernel.
            op (Function): Operation to apply.
        """
        super(Kernel, self).__init__()
        self.first = first
        self.second = second
        self.op = op

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        first = self.first(xi, xj, *args, **kwargs)
        second = self.second(xi, xj, *args, **kwargs)
        return self.op(first, second)


class RBFKernel(Kernel):

    """Radial-Basis Function Kernel."""

    def __init__(self, length_scale=None, sigma_s=None, eps=1e-6):
        """Constructs an RBFKernel.

        Args:
            length_scale (Tensor): Length scale.
            sigma_s (Tensor): Signal standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        self.length_scale = torch.nn.Parameter(
            torch.randn(1) if length_scale is None else length_scale)
        self.sigma_s = torch.nn.Parameter(
            torch.randn(1) if sigma_s is None else sigma_s)
        self._eps = eps

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        length_scale = (self.length_scale**-2).clamp(self._eps, 1e5)
        var_s = (self.sigma_s**2).clamp(self._eps, 1e5)

        M = torch.eye(xi.shape[1]) * length_scale
        dist = mahalanobis_squared(xi, xj, M)
        return var_s * (-0.5 * dist).exp()


class WhiteNoiseKernel(Kernel):

    """White noise kernel."""

    def __init__(self, sigma_n=None, eps=1e-6):
        """Constructs a WhiteNoiseKernel.

        Args:
            sigma_n (Tensor): Noise standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        self.sigma_n = torch.nn.Parameter(
            torch.randn(1) if sigma_n is None else sigma_n)
        self._eps = eps

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.

        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.

        Returns:
            Covariance (Tensor).
        """
        var_n = (self.sigma_n**2).clamp(self._eps, 1e5)
        return var_n


def mahalanobis_squared(xi, xj, VI=None):
    """Computes the pair-wise squared mahalanobis distance matrix as:

        (xi - xj)^T V^-1 (xi - xj)

    Args:
        xi (Tensor): xi input matrix.
        xj (Tensor): xj input matrix.
        VI (Tensor): The inverse of the covariance matrix, default: identity
            matrix.

    Returns:
        Weighted matrix of all pair-wise distances (Tensor).
    """
    if VI is None:
        xi_VI = xi
        xj_VI = xj
    else:
        xi_VI = xi.mm(VI)
        xj_VI = xj.mm(VI)

    D = (xi_VI * xi).sum(dim=-1).reshape(-1, 1) \
      + (xj_VI * xj).sum(dim=-1).reshape(1, -1) \
      - 2 * xi_VI.mm(xj.t())
    return D
