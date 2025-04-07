import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def loss_wrapper(core_fn):
    def wrapped(x, param=None, reduction="mean"):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        if param is None:
            param = 1.0
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=dtype, device=device)
        else:
            param = param.to(device=device, dtype=dtype)

        losses = core_fn(x, param)  # must return shape (n,)

        if reduction == "mean":
            return losses.mean()
        elif reduction == "sum":
            return losses.sum()
        elif reduction == "none":
            return losses
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    return wrapped

def smoothed_simplex_core(x, sigma):
    n, p = x.shape
    sigma1 = sigma
    sigma2 = sigma * math.sqrt(p)

    def psi(z, s):
        scaled = z / s
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        phi = normal.log_prob(-scaled).exp()
        Phi = normal.cdf(-scaled)
        return s * phi + z * (1 - Phi)

    neg_part = psi(-x, sigma1).sum(dim=1)
    sum_constraint = x.sum(dim=1) - 1
    sum_part = psi(sum_constraint, sigma2)

    return neg_part + sum_part  # shape (n,)

def quadratic_sum_penalty_core(x, rho):
    norm_squared = torch.sum(x**2, dim=1)
    sum_squared = torch.sum(x, dim=1) ** 2
    return 0.5 * norm_squared - rho * sum_squared  # shape (n,)

def relu_simplex_core(x, _param=None):
    """
    ReLU version of the smoothed simplex loss.

    Args:
        x: (n, p) torch tensor
        _param: ignored, included for API compatibility

    Returns:
        Tensor of shape (n,)
    """
    neg_part = torch.relu(-x).sum(dim=1)                 # penalize negative values
    sum_constraint = x.sum(dim=1) - 1
    sum_part = torch.relu(sum_constraint)                # penalize sum > 1
    return neg_part + sum_part                           # shape (n,)
