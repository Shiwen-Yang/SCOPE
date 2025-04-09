import torch
import math
from torch.func import jacrev, hessian, vmap


def batched_grad_hess_functorch(f, x, param):
    """
    Computes batched gradients and Hessians for a batch of inputs x using functorch.

    Args:
        f: a loss function of the form f(x, param), where x has shape (1, p)
        x: input tensor of shape (N, p), must be on the target device
        param: parameter to be passed to f

    Returns:
        grads: Tensor of shape (N, p)
        hessians: Tensor of shape (N, p, p)
    """
    device = x.device
    dtype = x.dtype

    def single_loss(xi):
        return f(xi.unsqueeze(0).to(device=device, dtype=dtype), param)

    grad_fn = jacrev(single_loss)
    hess_fn = hessian(single_loss)

    grads = vmap(grad_fn)(x)
    hessians = vmap(hess_fn)(x)

    return grads.to(device), hessians.to(device)


def skew_basis(p, device = "cuda", dtype = torch.float32):
    basis = []
    for i in range(p):
        for j in range(i + 1, p):
            B = torch.zeros(p, p, device=device, dtype=dtype)
            B[i, j] = 1 / math.sqrt(2)
            B[j, i] = -1 / math.sqrt(2)
            basis.append(B)
    return torch.stack(basis)  # (d, p, p)

def construct_M_loop(data, loss_fn, param):
    device, dtype = data.device, data.dtype
    N, p = data.shape
    B_list = skew_basis(p, device=device, dtype=dtype)
    d = len(B_list)

    grads, hessians = batched_grad_hess_functorch(loss_fn, data, param)  # (N, p), (N, p, p)

    M = torch.zeros((d, d), dtype=dtype, device=device)

    for i, Bi in enumerate(B_list):
        for j, Bj in enumerate(B_list):
            term1 = 0.0
            term2 = 0.0
            for n in range(N):
                y_n = data[n]
                g_n = grads[n]
                H_n = hessians[n]

                Bi_y = Bi @ y_n
                Bj_y = Bj @ y_n

                t1 = Bi_y @ H_n @ Bj_y  # scalar
                t2 = g_n @ Bi.T @ Bj @ y_n  # scalar

                term1 += t1
                term2 += t2

            M[i, j] = (term1 - term2) / N

    return M


if __name__ == "__main__":

    from torch.distributions import Dirichlet
    from src import loss_functions as LF

    alpha = torch.tensor([[10, 1, 1], [1, 10, 1]], dtype= torch.float64)
    # alpha = torch.tensor([[5, 1]], dtype= torch.float64)
    n= 20000
    K, p = alpha.shape
    torch.manual_seed(5)
    dir = Dirichlet(alpha)
    X = dir.sample((n // K,)).transpose(0, 1).reshape(n, p)[:, :2]

    smoothed_simplex_loss = LF.loss_wrapper(LF.smoothed_simplex_core)
    print(construct_M_loop(X, smoothed_simplex_loss, 0.2))