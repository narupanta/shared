import torch
import gpytorch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import * 
from gpytorch.mlls import SumMarginalLogLikelihood

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


class TensorBasisGaussianProcessModel() :
    def __init__(self, train_x, train_y):
        l1, l2, l3 = gpytorch.likelihoods.GaussianLikelihood(), gpytorch.likelihoods.GaussianLikelihood(), gpytorch.likelihoods.GaussianLikelihood()
        self.c1 = ExactGPModel(train_x, train_y[:, 0], l1).to(train_x.device)
        self.c2 = ExactGPModel(train_x, train_y[:, 1], l2).to(train_x.device)
        self.c3 = ExactGPModel(train_x, train_y[:, 2], l3).to(train_x.device)
        self.model = gpytorch.models.IndependentModelList(self.c1, self.c2, self.c3)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(self.c1.likelihood, self.c2.likelihood, self.c3.likelihood)
    def optimize_hyperparameters(self, n_iterations, optimizer) :
        self.model.train()
        self.likelihood.train()

        # Use a realistic learning rate for Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1)  

        # Scheduler reduces lr by factor 0.5 after 20 epochs of no improvement
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        for i in range(n_iterations):
            optimizer.zero_grad()
            mll = SumMarginalLogLikelihood(self.likelihood, self.model)
            output = self.model(*self.model.train_inputs)
            loss = -mll(output, self.model.train_targets)
            loss.backward()

            optimizer.step()
            scheduler.step(loss)
            #### write better log ####
            if i % 1 == 0 or i == n_iterations - 1:
                print(f"Iter {i+1}/{n_iterations} - Loss: {loss.item():.6f} - lr: {optimizer.param_groups[0]['lr']:.2e}")
        # return model, likelihood
    
    def save_model() :
        return
    def load_model() :
        return

    def predict_coeffs(self, invariants) :
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(*self.model(invariants, invariants, invariants))
        return predictions
    def predict_cauchy_stress(self, deformation_gradient) :
        b = B_func(deformation_gradient).detach()
        invariants = torch.stack([I1_func(b), I2_func(b), I3_func(b)]).T
        coeffs = self.predict_coeffs(invariants)

        c_means = [c.mean for c in coeffs]
        c_stds  = [c.variance.sqrt() for c in coeffs]

        sigma_mean = (
            c_means[0][:, None, None] * torch.eye(3) +
            c_means[1][:, None, None] * b +
            c_means[2][:, None, None] * (b @ b)
        )

        # Propagate uncertainty linearly (approx)
        sigma_std = (
            c_stds[0][:, None, None] * torch.eye(3).abs() +
            c_stds[1][:, None, None] * b.abs() +
            c_stds[2][:, None, None] * (b @ b).abs()
        )
        return sigma_mean, sigma_std
    