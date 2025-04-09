import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider




def smoothed_relu(z, sigma):
    """
    Smooth approximation of ReLU using Gaussian mollifier.

    Args:
        z: tensor
        sigma: smoothing parameter (scalar or tensor broadcastable to z)

    Returns:
        Smoothed ReLU approximation, same shape as z
    """
    sqrt_2 = math.sqrt(2)
    sqrt_2pi = math.sqrt(2 * math.pi)

    scaled = z / sigma
    phi = torch.exp(-0.5 * scaled**2) / sqrt_2pi
    Phi = 0.5 * (1 + torch.erf(scaled / sqrt_2))

    return sigma * phi + z * Phi

def mollified_relu_simplex_core(x, sigma):
    """
    Smooth version of the ReLU simplex loss using smoothed ReLU.

    Args:
        x: (n, p) tensor
        sigma: smoothing parameter (float or tensor)

    Returns:
        Tensor of shape (n,) with smoothed loss values
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    n, p = x.shape
    sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    p = torch.tensor(p, device=x.device)
    
    # Negative entries penalty
    neg_part = smoothed_relu(-x, sigma).sum(dim=1)

    # Sum constraint penalty
    sum_constraint = torch.sum(x) - 1  # penalize when > 1
    sum_part = smoothed_relu(sum_constraint, sigma*torch.sqrt(p))

    return neg_part + sum_part

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



