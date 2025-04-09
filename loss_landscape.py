import torch
from torch.distributions import Dirichlet
import src.loss_functions as LF
from src.utils import LossExplorer, plot_explorer_landscape_with_data


alpha = torch.tensor([[10, 1, 1], [1, 10, 1]], dtype= torch.float64)
# alpha = torch.tensor([[5, 1]], dtype= torch.float64)
n= 2000
K, p = alpha.shape
torch.manual_seed(5)
dir = Dirichlet(alpha)
X = dir.sample((n // K,)).transpose(0, 1).reshape(n, p)[:, :2]

smoothed_simplex_loss = LF.loss_wrapper(LF.mollified_relu_simplex_core)
quadratic_sum_penalty_loss = LF.loss_wrapper(LF.quadratic_sum_penalty_core)
relu_simplex_loss = LF.loss_wrapper(LF.relu_simplex_core)
       
        
if __name__ == "__main__":

    # Define parameter sweep (e.g., sigma or rho)
    param_vals = torch.linspace(-0.1, 0.1, 40)

    # Create LossExplorer
    explorer = LossExplorer(
        data=X,
        loss_fn=smoothed_simplex_loss,
        # loss_fn=LF.quadratic_sum_penalty_loss,
        param_vals=param_vals,
        param_label=r"$\sigma$"
    )

    # Launch interactive visualization
    plot_explorer_landscape_with_data(explorer)