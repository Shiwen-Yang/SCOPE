import torch
import math

from torch.distributions import Normal



def compute_Gpp_basis(g_prime, g_double_prime, B1, B2, Y):
    """
    Compute empirical estimate of E[(B1 y)^T g''(y) B2 y - g'(y)^T B1^T B2 y]

    Args:
        g_prime: function (N x p) -> (N x p), gradient function
        g_double_prime: function (N x p) -> (N x p x p), Hessian function
        B1: skew-symmetric matrix (p x p)
        B2: skew-symmetric matrix (p x p)
        Y: data matrix (N x p)

    Returns:
        Scalar: empirical estimate of second derivative contribution
    """
    B1 = B1.float()
    B2 = B2.float()
    Y = Y.float()

    B1Y = Y @ B1.T  # (N x p)
    B2Y = Y @ B2.T  # (N x p)

    # Hessian term: (B1 y)^T H (B2 y)
    H = g_double_prime(Y)        # (N x p x p)
    B2Y_exp = B2Y.unsqueeze(2)   # (N x p x 1)
    HB2Y = torch.bmm(H, B2Y_exp) # (N x p x 1)
    term1 = torch.bmm(B1Y.unsqueeze(1), HB2Y).squeeze()  # (N,)
    
    # Gradient term: g'(y)^T B1^T B2 y
    B1B2Y = Y @ (B2.T @ B1).T  # (N x p)
    gY = g_prime(Y)            # (N x p)
    term2 = (gY * B1B2Y).sum(dim=1)  # (N,)

    return (term1 - term2).mean().item()


def mollified_relu_grad(Y, sigma=0.02):
    """
    Compute gradient of mollified ReLU loss w.r.t. Y.
    Y: (N x p), sigma: scalar
    Returns: (N x p) gradient matrix
    """
    normal = Normal(0, 1)
    N, p = Y.shape

    # Negative term gradient
    x = -Y
    grad_neg = normal.cdf(x / sigma)

    # Sum constraint term
    w = Y.sum(dim=1, keepdim=True) - 1  # shape: (N x 1)
    grad_sum = normal.cdf(w / (sigma * p**0.5))  # shape: (N x 1)
    grad_sum_expanded = grad_sum.expand(-1, p)

    return grad_neg + grad_sum_expanded


def mollified_relu_hess(Y, sigma=0.02):
    """
    Compute Hessian of mollified ReLU loss w.r.t. Y.
    Returns (N x p x p) Hessians per sample.
    """
    normal = Normal(0, 1)
    N, p = Y.shape

    # Negative term Hessian (diagonal)
    x = -Y
    diag_vals = normal.log_prob(x / sigma).exp() / sigma  # shape: (N x p)
    H_neg = torch.diag_embed(diag_vals)  # (N x p x p)

    # Sum constraint Hessian (rank-1 update)
    w = Y.sum(dim=1, keepdim=True) - 1  # (N x 1)
    phi_vals = normal.log_prob(w / (sigma * p**0.5)).exp() / (sigma * p**0.5)  # (N x 1)
    ones = torch.ones((N, p, 1), device=Y.device)
    H_sum = phi_vals.view(-1, 1, 1) * torch.bmm(ones, ones.transpose(1, 2))  # (N x p x p)

    return H_neg + H_sum

def skew_basis(p, device = "cuda", dtype = torch.float32):
    basis = []
    for i in range(p):
        for j in range(i + 1, p):
            B = torch.zeros(p, p, device=device, dtype=dtype)
            B[i, j] = 1 / math.sqrt(2)
            B[j, i] = -1 / math.sqrt(2)
            basis.append(B)
    return torch.stack(basis)  # (d, p, p)

def construct_M_from_basis(g_prime, g_double_prime, B, Y):
    """
    Construct M matrix using basis elements B and gradient/Hessian functions.

    Args:
        g_prime: grad function (N x p) -> (N x p)
        g_double_prime: hessian function (N x p) -> (N x p x p)
        B: tensor of skew basis matrices (d x p x p)
        Y: data matrix (N x p)

    Returns:
        M: tensor of shape (d x d)
    """
    d = B.shape[0]
    M = torch.zeros(d, d, device=Y.device, dtype=Y.dtype)

    for i in range(d):
        for j in range(d):
            M[i, j] = compute_Gpp_basis(g_prime, g_double_prime, B[i], B[j], Y)

    return M


if __name__ == "__main__":

    from torch.distributions import Dirichlet
    from src import loss_functions as LF


    alpha = torch.ones(4, 4, dtype= torch.float32) + torch.eye(4, dtype= torch.float32)*9
    # alpha = torch.tensor([[10, 1, 1, 1], [1, 10, 1], [1, 1, 10]], dtype= torch.float64)
    # alpha = torch.tensor([[5, 1]], dtype= torch.float64)
    n= 30000
    K, p = alpha.shape
    torch.manual_seed(5)
    dir = Dirichlet(alpha)
    X = dir.sample((n // K,)).transpose(0, 1).reshape(n, p)[:, :4].to("cuda")
    
    B = skew_basis(p, "cuda")
    print(construct_M_from_basis(mollified_relu_grad, mollified_relu_hess, B, X))
