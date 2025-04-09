import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def loss_wrapper(core_fn, reduction = "mean"):
    def wrapped(x, param=None, reduction= reduction):
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
        elif reduction == "none":
            return losses
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    return wrapped

class LossFunctionWrapper:
    def __init__(self, loss_fn, reduction="mean", gradient_fn = None, hessian_fn = None):
        """
        A wrapper for loss function evaluation with optional manual gradient/Hessian computation.

        Args:
            loss_fn (callable): Function loss(x, param) → shape (n,)
            gradient_fn (callable, optional): grad(x, param, reduction) → shape (n, p) or (1, p)
            hessian_fn (callable, optional): hess(x, param, reduction) → shape (n, p, p) or (1, p, p)
            reduction (str): "mean" or "none"
        """
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn
        self.hessian_fn = hessian_fn
        self.reduction = reduction

    def evaluate(self, x, param, reduction=None):
        """
        Evaluate the loss function on input x.

        Args:
            x (Tensor): Input of shape (n, p) or (p,)
            param (float or Tensor): Additional parameter to the loss function
            reduction (str, optional): Overrides default reduction ("mean" or "none")

        Returns:
            Tensor: Scalar if reduction == "mean", otherwise tensor of shape (n,)
        """
        reduction = reduction if reduction is not None else self.reduction 
        loss_fn = loss_wrapper(self.loss_fn, reduction = reduction)
        return loss_fn(x, param, reduction = reduction)

    def gradient(self, x, param, reduction=None):
        """
        Compute the gradient of the loss function with respect to x.

        Args:
            x (Tensor): Input of shape (n, p) or (p,)
            param (float or Tensor): Additional parameter to the loss function
            reduction (str, optional): Overrides default reduction

        Returns:
            Tensor:
                - shape (1, p) if reduction == "mean"
                - shape (n, p) if reduction == "none"
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Converts shape (p,) → (1, p)

        reduction = reduction if reduction is not None else self.reduction
        
        if self.gradient_fn is not None:
            return self.gradient_fn(x, param, reduction)
        
        def scalar_loss(xi):
            return loss_wrapper(self.loss_fn, reduction="mean")(xi.unsqueeze(0), param)

        grad_fn = torch.func.grad(scalar_loss)
        gradient = torch.vmap(grad_fn)(x)

        if reduction == "mean":
            return gradient.mean(dim = 0, keepdim = True)
        else:
            return gradient

    def hessian(self, x, param, reduction=None):
        """
        Compute the Hessian of the loss with respect to x.

        Args:
            x (Tensor): Input of shape (n, p) or (p,)
            param (float or Tensor): Additional parameter to the loss function
            reduction (str, optional): Overrides default reduction

        Returns:
            Tensor:
                - shape (1, p, p) if reduction == "mean"
                - shape (n, p, p) if reduction == "none"
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Converts shape (p,) → (1, p)
        reduction = reduction if reduction is not None else self.reduction
        
        if self.hessian_fn is not None:
            return self.hessian_fn(x, param, reduction)
        
        def scalar_loss(xi):
            return loss_wrapper(self.loss_fn, reduction="mean")(xi.unsqueeze(0), param)

        hess_fn = torch.func.hessian(scalar_loss)  # returns (p, p) Hessian per sample
        hessians = torch.vmap(hess_fn)(x)     # shape (n, p, p)
        
        if reduction == "mean":
            return hessians.mean(dim=0, keepdim = True)  # shape (p, p)
        else:
            return hessians  # shape (n, p, p)
        


class RotationHandler:
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def get_rotation_matrix(self, theta):
        """
        Construct the 2D rotation matrix R(θ).
        Args:
            theta (float or torch scalar)
        Returns:
            R: shape (2, 2)
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=self.dtype, device=self.device)
        else:
            theta = theta.to(dtype=self.dtype, device=self.device)

        c, s = torch.cos(theta), torch.sin(theta)

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c])
        ], dim=0)

        return R  # shape (2, 2)

    def rotate(self, x, theta):
        """
        Rotate 2D vector(s) by angle theta.
        Args:
            x: shape (2,) or (n, 2)
            theta: float or torch scalar
        Returns:
            Rotated tensor, same shape as x
        """
        R = self.get_rotation_matrix(theta)

        if x.ndim == 1:
            return (R @ x).view(2)
        elif x.ndim == 2:
            return x @ R.T
        else:
            raise ValueError("x must be shape (2,) or (n, 2)")



class LossLandscapeCache:
    def __init__(self, explorer):
        """
        Args:
            explorer: LossExplorer instance (must have .data, .loss, .rotator)
        """
        self.explorer = explorer
        self.curves = {}      # param_idx -> (theta_vals, loss_vals)
        self.grids = {}       # param_idx -> (X, Y, Z)
        self.hessians = {}    # param_idx -> list of (theta, hess1, hess2)
    
    def compute_loss_curve(self, param, thetas):
        """
        Computes total loss for each theta.
        Args:
            param: scalar param (e.g., sigma or rho)
            thetas: tensor of shape (T,)
        Returns:
            tensor of shape (T,)
        """
        data = self.explorer.data  # (n, 2)
        loss_fn = self.explorer.loss.evaluate
        rotate = self.explorer.rotator.rotate

        loss_curve = []
        for theta in thetas:
            x_rot = rotate(data, theta)
            loss_val = loss_fn(x_rot, param)  # mean reduction
            loss_curve.append(loss_val.item())
        
        return torch.tensor(loss_curve)

    def compute_loss_grid(self, param, xlim=(-1.1,1.1), ylim=(-1.1,1.1), resolution=225):
        """
        Computes a 2D loss surface over a grid of (x1, x2) locations.
        Vectorized for speed.

        Args:
            param: scalar loss parameter
            xlim, ylim: grid range
            resolution: number of points per axis

        Returns:
            X, Y, Z (all shape (res, res)) on CPU
        """
        loss_fn = self.explorer.loss.evaluate
        device = self.explorer.device
        dtype = self.explorer.dtype

        # Create meshgrid on correct device
        x_vals = torch.linspace(xlim[0], xlim[1], resolution, device=device, dtype=dtype)
        y_vals = torch.linspace(ylim[0], ylim[1], resolution, device=device, dtype=dtype)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')

        flat_inputs = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (res*res, 2)

        # Vectorized batch loss (no loop!)
        with torch.no_grad():
            Z_flat = loss_fn(flat_inputs, param, reduction="none")  # shape (res*res,)

        Z = Z_flat.reshape(resolution, resolution).cpu()
        return X.cpu(), Y.cpu(), Z


    def precompute_all(self, param_vals, thetas, resolution=100):
        """
        Compute all curves and grids and store them.
        """
        for i, param in enumerate(param_vals):
            self.curves[i] = self.compute_loss_curve(param, thetas)
            self.grids[i] = self.compute_loss_grid(param, resolution=resolution)
            # Optionally, add hessian curve caching here later




class LossExplorer:
    def __init__(
        self,
        data,
        loss_fn,
        param_vals,
        param_label=r"$\rho$",
        reduction="mean",
        device=None,
        dtype=torch.float32
    ):
        """
        Master class for exploring 2D loss functions.

        Args:
            data (array-like): shape (n, 2)
            loss_fn (callable): takes x, param, reduction -> scalar or tensor
            param_vals (Tensor or list): 1D values to sweep over
            param_label (str): Label for the parameter
            reduction (str): Default reduction method
            device (str): "cpu" or "cuda"
            dtype (torch.dtype): float32 or float64
        """
        # Data
        if isinstance(data, torch.Tensor):
            self.data = data.to(dtype=dtype)
        else:
            self.data = torch.tensor(data, dtype=dtype)

        if self.data.ndim == 1:
            self.data = self.data.unsqueeze(0)
        self.n, self.p = self.data.shape
        assert self.p == 2, "LossExplorer currently supports 2D only."

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.data = self.data.to(self.device)

        # Loss + wrapper
        self.loss = LossFunctionWrapper(loss_fn, reduction=reduction)

        # Param sweeping
        if isinstance(param_vals, torch.Tensor):
            self.param_vals = param_vals.to(dtype=dtype, device=self.device)
        else:
            self.param_vals = torch.tensor(param_vals, dtype=dtype, device=self.device)

        self.param_label = param_label

        # Rotation
        self.rotator = RotationHandler(device=self.device, dtype=self.dtype)

        # Caching
        self.cache = LossLandscapeCache(self)

    def data_mean_and_cov(self):
        """
        Returns the empirical mean and covariance matrix of the data.
        """
        x = self.data  # (n, 2)
        mean = x.mean(dim=0)
        x_centered = x - mean
        cov = (x_centered.T @ x_centered) / (x.shape[0] - 1)
        return mean, cov
    
    def rotated_mean_and_cov(self, theta):
        """
        Rotate the empirical mean and covariance by theta.
        """
        mean, cov = self.data_mean_and_cov()
        R = self.rotator.get_rotation_matrix(theta)
        rotated_mean = R @ mean
        rotated_cov = R @ cov @ R.T
        return rotated_mean, rotated_cov
    
    def rotate_data(self, theta):
        """
        Rotate all input data by theta using the RotationHandler.
        Returns rotated data.
        """
        return self.rotator.rotate(self.data, theta)

    def evaluate_loss(self, theta, param):
        """
        Rotate data and evaluate loss at that configuration.
        """
        x_rot = self.rotate_data(theta)
        return self.loss.evaluate(x_rot, param)

    def compute_curve(self, param):
        """
        Compute loss-vs-theta curve at given param value.
        Returns tensor of shape (T,)
        """
        thetas = torch.linspace(-torch.pi/2, torch.pi/2, 180, device=self.device)
        return self.cache.compute_loss_curve(param, thetas)

    def compute_grid(self, param, resolution=100):
        """
        Compute 2D heatmap loss surface for given param value.
        """
        return self.cache.compute_loss_grid(param, resolution=resolution)

    def precompute_all(self, resolution=100):
        """
        Precompute all curves and grids over param_vals.
        """
        thetas = torch.linspace(-torch.pi/2, torch.pi/2, 180, device=self.device)
        self.cache.precompute_all(self.param_vals, thetas, resolution=resolution)
        
        
        
        
def plot_explorer_landscape_with_data(explorer, resolution=100):
    thetas = torch.linspace(-np.pi / 2, np.pi / 2, 180, device=explorer.device)
    param_vals = explorer.param_vals
    explorer.precompute_all(resolution=resolution)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)

    # Slider axes
    theta_slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
    rho_slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])

    # Sliders
    theta_slider = Slider(theta_slider_ax, r"$\theta$", -np.pi/2, np.pi/2, valinit=0)
    rho_slider = Slider(rho_slider_ax, explorer.param_label, 
                        valmin=param_vals.min().item(), 
                        valmax=param_vals.max().item(), 
                        valinit=param_vals[0].item(), 
                        valfmt="%.3f")

    # Initial plot
    X, Y, Z = explorer.cache.grids[0]
    im = ax1.contourf(X, Y, Z, levels=50, cmap="viridis")
    cb = fig.colorbar(im, ax=ax1)

    # Plot rotating data points
    rotated_data = explorer.rotate_data(0).cpu().numpy()
    data_dots, = ax1.plot(rotated_data[:, 0], rotated_data[:, 1], linestyle='none', marker='o',
                      color='red', markersize=2, alpha=0.7, label="Rotated Data", zorder=4)


    ax1.set_title("Loss Landscape with Rotated Data Points")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.legend()

    # Plot loss vs theta
    theta_vals_np = thetas.cpu().numpy()
    loss_line, = ax2.plot(theta_vals_np, explorer.cache.curves[0])
    red_vline = ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Loss vs. $\\theta$")
    ax2.set_xlabel("$\\theta$")
    ax2.set_ylabel("Loss")

    def update(val):
        theta = theta_slider.val
        rho_val = rho_slider.val
        pidx = torch.argmin(torch.abs(param_vals - rho_val)).item()
        param = param_vals[pidx]

        # Update loss landscape
        X, Y, Z = explorer.cache.grids[pidx]
        for coll in ax1.collections:
            coll.remove()
        ax1.contourf(X, Y, Z, levels=50, cmap="viridis")

        # Update rotated data points
        rotated_data = explorer.rotate_data(theta).cpu().numpy()
        data_dots.set_data(rotated_data[:, 0], rotated_data[:, 1])

        # Update loss curve
        loss_vals = explorer.cache.curves[pidx]
        loss_line.set_ydata(loss_vals)
        red_vline.set_xdata([theta])
        # Dynamically adjust y-axis for loss curve
        loss_min = loss_vals.min().item()
        loss_max = loss_vals.max().item()
        padding = 0.1 * (loss_max - loss_min) if (loss_max - loss_min) > 1e-6 else 1.0
        ax2.set_ylim(loss_min - padding, loss_max + padding)

        fig.canvas.draw_idle()

    theta_slider.on_changed(update)
    rho_slider.on_changed(update)
    update(0)

    plt.show()