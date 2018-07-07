"""Gaussian Process models."""

import torch
import warnings
import numpy as np


class GaussianProcess(torch.nn.Module):

    """Gaussian Process regressor.

    This is meant for multi-input, single-output functions.
    It still works for multi-output functions, but will share the same
    hyperparameters for each output. If that is not wanted, use
    `MultiGaussianProcess` instead.
    """

    def __init__(self,
                 kernel,
                 X,
                 Y,
                 sigma_n=None,
                 eps=1e-6,
                 reg=1e-5,
                 normalize_y=True):
        """Constructs a GaussianProcess.

        Args:
            kernel (Kernel): Kernel.
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            sigma_n (Tensor): Noise standard deviation.
            eps (float): Minimum bound for parameters.
            reg (float): Regularization term to guarantee the kernel matrix is
                positive definite.
            normalize_y (bool): Whether to normalize the outputs.
        """
        super(GaussianProcess, self).__init__()
        self.kernel = kernel
        self.sigma_n = torch.nn.Parameter(
            torch.randn(1) if sigma_n is None else sigma_n)
        self.reg = torch.nn.Parameter(torch.tensor(reg), requires_grad=False)

        self._eps = eps
        self._non_normalized_Y = Y

        if normalize_y:
            Y_mean = torch.mean(Y, dim=0)
            Y_variance = torch.std(Y, dim=0)
            Y = (Y - Y_mean) / Y_variance

        self._X = X
        self._Y = Y

        self._update_k()

    def _update_k(self):
        """Updates the K matrix."""
        X = self._X
        Y = self._Y

        # Compute K and guarantee it's positive definite.
        var_n = (self.sigma_n**2).clamp(self._eps, 1e5)
        K = self.kernel(X, X)
        K = (K + K.t()).mul(0.5)
        self._K = K + (self.reg + var_n) * torch.eye(X.shape[0])

        # Compute K's inverse and Cholesky factorization.
        # We can't use potri() to compute the inverse since it's derivative
        # isn't implemented yet.
        self._L = torch.potrf(self._K)
        self._K_inv = self._K.inverse()

    def update(self, X, Y):
        """Update the training data.

        Args:
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
        """
        self._X = X
        self._Y = Y
        self._update_k()

    def loss(self):
        """Computes the loss as the negative marginal log likelihood."""
        Y = self._Y
        self._update_k()
        K_inv = self._K_inv

        # Compute the log likelihood.
        log_likelihood_dims = -0.5 * Y.t().mm(K_inv.mm(Y)).sum(dim=0)
        log_likelihood_dims -= self._L.diag().log().sum()
        log_likelihood_dims -= self._L.shape[0] / 2.0 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(dim=-1)

        return -log_likelihood

    def forward(self,
                x,
                return_mean=True,
                return_covar=False,
                return_std=False,
                **kwargs):
        """Computes the GP estimate.

        Args:
            x (Tensor): Inputs.
            return_mean (bool): Whether to return the mean.
            return_covar (bool): Whether to return the covariance.
            return_std (bool): Whether to return the standard deviation.

        Returns:
            Tensor or tuple of Tensors.
        """
        X = self._X
        Y = self._Y
        K_inv = self._K_inv

        # Kernel functions.
        K_ss = self.kernel(x, x)
        K_s = self.kernel(x, X)

        # Compute mean.
        outputs = []
        if return_mean:
            # Non-normalized for scale.
            mean = K_s.mm(K_inv.mm(self._non_normalized_Y))
            outputs.append(mean)

        # Compute covariance/standard deviation.
        if return_covar or return_std:
            covar = K_ss - K_s.mm(K_inv.mm(K_s.t()))
            if return_covar:
                outputs.append(covar)
            if return_std:
                std = covar.diag().sqrt().reshape(-1, 1)
                outputs.append(std)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def fit(self,
            tol=1e-6,
            reg=1e-5,
            reg_factor=10.0,
            max_reg=1.0,
            max_iter=1000):
        """Fits the model.

        Args:
            tol (float): Tolerance.
            reg (float): Regularization term to guarantee the kernel matrix is
                positive definite.
            reg_factor (float): Regularization multiplicative factor.
            max_reg (float): Maximum regularization term.
            max_iter (int): Maximum number of iterations.

        Returns:
            Number of iterations.
        """
        opt = torch.optim.Adam(p for p in self.parameters() if p.requires_grad)
        self.reg = torch.nn.Parameter(torch.tensor(reg), requires_grad=False)

        while self.reg <= max_reg:
            try:
                curr_loss = np.inf
                n_iter = 0

                while n_iter < max_iter:
                    opt.zero_grad()

                    prev_loss = self.loss()
                    prev_loss.backward(retain_graph=True)
                    opt.step()

                    curr_loss = self.loss()
                    dloss = curr_loss - prev_loss
                    n_iter += 1
                    if dloss.abs() <= tol:
                        break

                return n_iter
            except RuntimeError:
                # Increase regularization term until it succeeds.
                self.reg *= reg_factor
                continue

        warnings.warn("exceeded maximum regularization: did not converge")


class MultiGaussianProcess(torch.nn.Module):

    """
    Layer of abstraction to estimate vector-valued functions with a separate
    Gaussian Process for each output dimension.

    This learns separate hyperparameters for each output dimension instead of
    a single set for all dimensions.
    """

    def __init__(self, kernels, X, Y, *args, **kwargs):
        """Constructs a MultiGaussianProcess.

        Args:
            kernels (list<Kernel>): List of kernels to use for each dimension.
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            *args, **kwargs: Additional positional and key-word arguments to
                each Gaussian Process constructor.
        """
        super(MultiGaussianProcess, self).__init__()
        [
            self.add_module(
                "process_{}".format(i),
                GaussianProcess(kernel, X, Y[:, i].reshape(-1, 1), *args,
                                **kwargs)) for i, kernel in enumerate(kernels)
        ]
        self._processes = [
            getattr(self, "process_{}".format(i)) for i in range(len(kernels))
        ]
        self._X = X
        self._Y = Y

    def update(self, X, Y):
        """Update the training data.

        Args:
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
        """
        self._X = X
        self._Y = Y
        for i, gp in enumerate(self.processes):
            gp.update(X, Y[:, i].reshape(-1, 1))

    def loss(self):
        """Computes the loss as the negative marginal log likelihood."""
        loss = torch.tensor(0.0)
        for gp in self._processes:
            loss += gp.loss()
        return loss

    def fit(self, *args, **kwargs):
        """Fits the model.

        Args:
            *args, **kwargs: Additional positional and key-word arguments to
                pass to each gaussian process's fit() function.

        Returns:
            Total number of iterations.
        """
        # Fitting them individually is faster than fitting them together as
        # they are completely independent of each other.
        iters = 0
        for gp in self._processes:
            iters += gp.fit(*args, **kwargs)
        return iters

    def forward(self, x, *args, **kwargs):
        """Computes the GP estimate.

        Args:
            x (Tensor): Inputs.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to each gaussian process's forward() function.

        Returns:
            Tensor or tuple of Tensors.
        """
        outputs = np.array([gp(x, *args, **kwargs) for gp in self._processes])

        if outputs.ndim > 1:
            outputs = [
                torch.cat(tuple(outputs[:, i]), dim=-1)
                for i in range(outputs.shape[1])
            ]
        else:
            outputs = torch.cat(tuple(outputs), dim=-1)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
