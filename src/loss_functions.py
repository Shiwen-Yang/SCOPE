import torch
import math
from torch.distributions import Normal


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
    sum_constraint = torch.sum(x, dim = 1) - 1  # penalize when > 1
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




def mollified_relu_grad(Y, sigma=0.02, reduction = None):
    """
    Compute gradient of mollified ReLU loss w.r.t. Y.
    Y: (N x p), sigma: scalar
    Returns: (N x p) gradient matrix
    """
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim = 0) # Converts shape (p, ) → (1, p)
    normal = Normal(0, 1)
    N, p = Y.shape

    # Negative term gradient (for smoothed_relu(-x_j))
    grad_neg = -normal.cdf(-Y / sigma)  # shape: (N x p)

    # Sum constraint gradient (for smoothed_relu(sum(x) - 1))
    w = Y.sum(dim=1, keepdim=True) - 1  # shape: (N x 1)
    grad_sum = normal.cdf(w / (sigma * p**0.5))  # shape: (N x 1)
    grad_sum_expanded = grad_sum.expand(-1, p)

    gradient_val = grad_neg + grad_sum_expanded
    
    if reduction == "mean":
        return gradient_val.mean(dim = 0, keepdim=True)
    else:
        return gradient_val


def mollified_relu_hess(Y, sigma=0.02, reduction = None):
    """
    Compute batched Hessian of mollified ReLU simplex loss for a batch of inputs.

    Args:
        Y (Tensor): Input tensor of shape (n, p)
        sigma (float): Smoothing parameter

    Returns:
        Tensor of shape (n, p, p): Hessians for each sample
    """
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim = 0) # Converts shape (p, ) → (1, p)
    n, p = Y.shape
    device = Y.device
    dtype = Y.dtype
    normal = Normal(0, 1)

    # Diagonal term for negative entries
    term_1_vals = normal.log_prob(-Y / sigma).exp() / sigma  # (n, p)
    H_neg = torch.diag_embed(term_1_vals)  # (n, p, p)

    # Rank-one outer product for sum constraint
    sum_constraint = Y.sum(dim=1) - 1  # (n,)
    scaling = math.sqrt(p) * sigma
    term_2_vals = normal.log_prob(sum_constraint / scaling).exp() / scaling  # (n,)

    ones = torch.ones((n, p, 1), device=device, dtype=dtype)
    H_sum = term_2_vals.view(n, 1, 1) * torch.bmm(ones, ones.transpose(1, 2))  # (n, p, p)

    hessian_val = H_neg + H_sum
    
    if reduction == "mean":
        return hessian_val.mean(dim = 0, keepdim=True)
    else:
        return hessian_val