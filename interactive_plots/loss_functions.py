import torch
import math

def smoothed_simplex_loss(x, rho):
    """
    Smoothed simplex loss function.

    Args:
        x (torch.Tensor): vector in R^p (i.e., V^T y)
        sigma (float or scalar): smoothing parameter

    Returns:
        torch.Tensor: scalar loss value
    """
    p = x.shape[0]
    if not isinstance(rho, torch.Tensor):
        sigma = torch.tensor(rho, dtype=x.dtype)
    else:
        sigma = rho.clone().detach().to(dtype=x.dtype)
    sigma1 = sigma
    sigma2 = sigma * math.sqrt(p)

    # Helper: smoothed ReLU
    def psi(z, s):
        u = z / s
        phi = torch.exp(-0.5 * u**2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + torch.erf(u / math.sqrt(2)))
        return s * phi + z * Phi

    # Negativity penalty
    neg_part = psi(-x, sigma1).sum()

    # Sum-to-1 penalty
    s = x.sum() - 1
    sum_part = psi(s, sigma2)

    return neg_part + sum_part


def example_square_loss(x, rho):
    return 0.5 * torch.dot(x, x) - rho * torch.sum(x)**2