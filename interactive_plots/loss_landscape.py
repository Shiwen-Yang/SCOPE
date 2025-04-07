import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from functools import partial
import loss_functions as LF

def rotate_vector_torch(theta, x):
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=x.dtype)
    else:
        theta = theta.clone().detach().to(dtype=x.dtype)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)]
    ], dtype=x.dtype)
    return R @ x

def compute_second_derivative_components(y, theta, loss_func):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    A = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    Ry = R @ y
    ARy = A @ Ry
    A2Ry = A @ A @ Ry
    H = torch.autograd.functional.hessian(lambda x: loss_func(x), Ry)
    Ry_detached = Ry.detach().clone().requires_grad_(True)
    g = loss_func(Ry_detached)
    g.backward()
    grad = Ry_detached.grad
    return torch.dot(ARy, H @ ARy).item(), torch.dot(grad, A2Ry).item()

def compute_loss_grid(loss_func, rho, X, Y):
    XY = np.stack([X, Y], axis=-1).reshape(-1, 2)
    Z = np.array([
        loss_func(torch.tensor(xy, dtype=torch.float32), rho).item()
        for xy in XY
    ])
    return Z.reshape(X.shape)

# def compute_theta_loss_and_hessian_curves(loss_func, y1, y2, param_vals, thetas):
#     loss_curves, hess1_curves, hess2_curves = [], [], []
#     for rho in param_vals:
#         f = partial(loss_func, rho=rho)
#         loss_theta, hess1_theta, hess2_theta = [], [], []
#         for theta in thetas:
#             y1_rot = rotate_vector_torch(theta, y1)
#             y2_rot = rotate_vector_torch(theta, y2)
#             loss_theta.append(f(y1_rot).item() + f(y2_rot).item())
#             t1y1, t2y1 = compute_second_derivative_components(y1, theta, f)
#             t1y2, t2y2 = compute_second_derivative_components(y2, theta, f)
#             hess1_theta.append(t1y1 + t1y2)
#             hess2_theta.append(t2y1 + t2y2)
#         loss_curves.append(torch.tensor(loss_theta))
#         hess1_curves.append(torch.tensor(hess1_theta))
#         hess2_curves.append(torch.tensor(hess2_theta))
#     return loss_curves, hess1_curves, hess2_curves

def compute_theta_loss_curves(loss_func, y1, y2, param_vals, thetas):
    loss_curves = []
    for rho in param_vals:
        f = partial(loss_func, rho=rho)
        loss_theta = []
        for theta in thetas:
            y1_rot = rotate_vector_torch(theta, y1)
            y2_rot = rotate_vector_torch(theta, y2)
            loss_theta.append(f(y1_rot).item() + f(y2_rot).item())
        loss_curves.append(torch.tensor(loss_theta))
    return loss_curves

def interactive_plot_with_landscape(loss_func, y, param_range=(-0.1, 0.1), param_label=r'$\rho$', N_param=30, M_theta=180):
    y1 = torch.tensor([y[0], y[1]], dtype=torch.float32)
    y2 = torch.tensor([y[1], y[0]], dtype=torch.float32)
    thetas = torch.linspace(-np.pi / 2, np.pi / 2, M_theta)
    param_vals = torch.linspace(param_range[0], param_range[1], N_param)
    x_vals = torch.linspace(-2, 2, 100)
    y_vals = torch.linspace(-2, 2, 100)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    X_np, Y_np = X.numpy(), Y.numpy()

    print("Precomputing...")
    loss_cache = compute_theta_loss_curves(loss_func, y1, y2, param_vals, thetas)
    all_losses = torch.cat(loss_cache)
    
    loss_min, loss_max = all_losses.min().item(), all_losses.max().item()

    landscape_cache = [compute_loss_grid(loss_func, rho.item(), X_np, Y_np) for rho in param_vals]

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, height_ratios=[20, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_right = ax2.twinx()
    slider_ax1 = fig.add_subplot(gs[1, 0])
    slider_ax2 = fig.add_subplot(gs[1, 1])

    theta_vals_np = thetas.numpy()
    loss_line, = ax2.plot(theta_vals_np, loss_cache[0], label="Total Loss")
    red_vline = ax2.axvline(0, color='red', linestyle='--')

    im = ax1.contourf(X_np, Y_np, landscape_cache[0], levels=50, cmap='viridis')
    cb = fig.colorbar(im, ax=ax1)
    pt1, = ax1.plot([], [], 'ro')
    pt2, = ax1.plot([], [], 'bo')
    ax1.set_title("Loss Landscape")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    # ax2.set_title("Loss + Second Derivative Terms vs $\\theta$")
    ax2.set_title("Loss vs $\\theta$")
    ax2.set_xlabel("$\\theta$")
    ax2.set_ylabel("Loss")
    # ax2_right.set_ylabel("Hessian Terms")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)

    theta_slider = Slider(slider_ax1, label=" ", valmin=-np.pi/2, valmax=np.pi/2, valinit=0, valfmt="%.0s")
    param_slider = Slider(slider_ax2, label=" ", valmin=0, valmax=N_param-1, valinit=N_param//2, valstep=1, valfmt="%.0s")

    def update(val):
        theta = theta_slider.val
        pidx = int(param_slider.val)
        param_slider.label.set_text(rf"{param_label} = {param_vals[pidx].item():.3f}")
        theta_slider.label.set_text(rf"$\theta$ = {theta:.2f}")

        Z = landscape_cache[pidx]
        ax1.clear()
        ax1.contourf(X_np, Y_np, Z, levels=50, cmap='viridis')
        y1_rot = rotate_vector_torch(theta, y1).numpy()
        y2_rot = rotate_vector_torch(theta, y2).numpy()
        ax1.plot(*y1_rot, 'ro')
        ax1.plot(*y2_rot, 'bo')
        ax1.set_title("Loss Landscape")
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")

        loss_line.set_ydata(loss_cache[pidx])
        padding = 0.1 * (loss_max - loss_min)
        ax2.set_ylim(loss_min - padding, loss_max + padding)

        red_vline.set_xdata([theta])
        fig.canvas.draw_idle()

    theta_slider.on_changed(update)
    param_slider.on_changed(update)
    update(0)
    plt.tight_layout()
    plt.show()


def example_loss(x, rho):
    return 0.5 * torch.dot(x, x) - rho * torch.sum(x)**2

# Run this script
if __name__ == "__main__":
    interactive_plot_with_landscape(
        loss_func=LF.smoothed_simplex_loss,
        y=(0.9, 0.1),
        param_range=(0, 0.5),
        param_label=r"$\rho$",
        N_param=30,
        M_theta=180
    )
