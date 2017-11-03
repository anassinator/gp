"""Optimization methods."""
import theano
import numpy as np
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b


def optimize(gp, params=None, optimizer=fmin_l_bfgs_b, bound=True,
             *args, **kwargs):
    """Optimizes a gaussian process.

    Args:
        gp (GaussianProcess): Gaussian process to optimize.
        params (list<SharedVariable>): List of hyperparameters to optimize.
            Default: All of them.
        optimizer (callable): Optimization method with the following signature:
            (loss_f: (inputs) -> loss,
             x0: inputs,
             d_loss_wrt_params_f: (inputs) -> derivative of loss,
             params,
             *args, **kwargs) -> optimized parameters.
        bound (bool): Constrain hyperparaters within bounds.
        *args, **kwargs: Additional positional and key-word arguments to pass
            to the optimizer.
    """
    if params is None:
        params = gp.hyperparameters

    inputs = [p.type() for p in params]
    x0 = [p.get_value() for p in params]

    givens = {params[i]: inputs[i] for i in range(len(params))}
    givens.update({gp.X: gp.X_train, gp.Y: gp.Y_train})

    def as_function(v):
        f = theano.function(inputs, v, givens=givens)
        return lambda x: np.array(f(*x))

    loss = -gp.log_likelihood
    loss_f = as_function([loss])
    d_loss_wrt_params_f = as_function(T.grad(loss, params))

    if bound:
        kwargs["bounds"] = [
            gp.bounds[i] for i, p in enumerate(gp.hyperparameters)
            if p in set(params)
        ]

    opt = optimizer(loss_f, x0, d_loss_wrt_params_f, *args, **kwargs)
    if isinstance(opt, tuple):
        opt = opt[0]

    for i, p in enumerate(params):
        p.set_value(opt[i])
