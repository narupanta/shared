import torch 
import gpytorch
from matplotlib import pyplot as plt
from core.utils import *
from core.datasetclass import RandomKnownModelGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gpytorch.mlls import SumMarginalLogLikelihood
from core.TensorBasisGaussianProcessModel import TensorBasisGaussianProcessModel 
def log_model_parameters(model):
    total_params = 0
    total_trainable = 0
    print("===== Model Parameters =====")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            total_trainable += num_params
        print(f"{name}: {param.size()} | params={num_params} | requires_grad={param.requires_grad}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {total_trainable}")
    print("============================")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    # --- Select device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Select Dataset ---
    n_samples = 2000
    gamma_range = (0.2, 4.0)
    generator = RandomKnownModelGenerator(n_samples, gamma_range, 0.0, None)
    F_train = generator.get_F()

    invariants = generator.get_invariants(F_train)
    sigma = generator.get_cauchy_stress(F_train)
    coeffs = generator.get_coeffs(F_train)


    model = TensorBasisGaussianProcessModel(invariants.to(device), coeffs.to(device))
    stress_pred_mean, stress_pred_std = model.optimize_hyperparameters(500, None)
