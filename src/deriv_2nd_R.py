import torch
import math
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import LossFunctionWrapper

def compute_Gpp_basis(loss_core, B1, B2, Y, sigma = 0.02, loss_core_grad = None, loss_core_hessian = None):
    """
    Compute empirical estimate of E[(B1 y)^T g''(y) B2 y - g'(y)^T B1^T B2 y]

    Args:
        loss_core_grad: function (N x p) -> (N x p), gradient function
        loss_core_hessian: function (N x p) -> (N x p x p), Hessian function
        B1: skew-symmetric matrix (p x p)
        B2: skew-symmetric matrix (p x p)
        Y: data matrix (N x p)

    Returns:
        Scalar: empirical estimate of second derivative contribution
    """
    B1 = B1.float()
    B2 = B2.float()
    Y = Y.float()
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim = 0)
        
    g = LossFunctionWrapper(loss_core, gradient_fn=loss_core_grad, hessian_fn=loss_core_hessian)

    B1Y = Y @ B1.T  # (N x p)
    B2Y = Y @ B2.T  # (N x p)

    # Hessian term: (B1 y)^T H (B2 y)
    H = g.hessian(Y, sigma, "none")        # (N x p x p)
    B2Y_exp = B2Y.unsqueeze(2)   # (N x p x 1)
    HB2Y = torch.bmm(H, B2Y_exp) # (N x p x 1)
    term1 = torch.bmm(B1Y.unsqueeze(1), HB2Y).squeeze()  # (N,)
    
    # Gradient term: g'(y)^T B1^T B2 y
    B1B2Y = Y @ (B2.T @ B1).T  # (N x p)
    gY = g.gradient(Y, sigma, "none")            # (N x p)
    term2 = (gY * B1B2Y).sum(dim=1)  # (N,)

    return (term1 - term2).mean().item()


def skew_basis(p, device = "cuda", dtype = torch.float32):
    basis = []
    for i in range(p):
        for j in range(i + 1, p):
            B = torch.zeros(p, p, device=device, dtype=dtype)
            B[i, j] = 1 / math.sqrt(2)
            B[j, i] = -1 / math.sqrt(2)
            basis.append(B)
    return torch.stack(basis)  # (d, p, p)

def construct_M_from_basis(loss_core, Y, sigma, loss_core_grad = None, loss_core_hessian = None):
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
    p = Y.shape[1]
    Basis = skew_basis(p, device = Y.device, dtype=Y.dtype)
    d = Basis.shape[0]
    M = torch.zeros(d, d, device=Y.device, dtype=Y.dtype)
    total = d * (d + 1) // 2  # number of (i,j) pairs where j ≥ i
    count = 0
    for i in range(d):
        for j in range(i, d):
            M[i, j] = compute_Gpp_basis(loss_core=loss_core, 
                                        B1=Basis[i], 
                                        B2=Basis[j], 
                                        Y=Y,
                                        sigma=sigma,
                                        loss_core_grad=loss_core_grad,
                                        loss_core_hessian=loss_core_hessian)
            M[j, i] = M[i, j]
            count += 1
            sys.stdout.write(f"\rCurrently at {100*count/total:.2f}%      ")
            sys.stdout.flush()
    return M



def plot_M_heatmap(M_vis: torch.Tensor, show_labels: bool = True):
    d = M_vis.shape[0]
    p = int((1 + math.sqrt(1 + 8 * d)) / 2)
    basis_labels = [f"({i},{j})" for i in range(p) for j in range(i+1, p)]
    M_vis_np = M_vis.cpu().detach().numpy()

    tick_labels = basis_labels if show_labels else False

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        M_vis_np,
        cmap='coolwarm',
        center=0,
        annot=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar=True
    )

    plt.title(f"Heatmap of M with Edge-Pair Labels (p={p})", fontsize=24, pad=20)

    if show_labels:
        plt.xticks(rotation=90, fontsize=18)
        plt.yticks(rotation=0, fontsize=18)
    else:
        plt.xticks([])
        plt.yticks([])

    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Interaction Value", fontsize=18)
    colorbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()
