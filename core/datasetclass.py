import os
import json
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
jax.config.update("jax_enable_x64", True)
from scipy.optimize import root
from .material_models import BaseMaterialModel, get_material
from .utils import (
    B_func,
    I1_func,
    I2_func,
    I3_func,
    J_func,
    solve_for_coefficients_batched,
    generate_random_F_plane_stress,
    deformation_gradient_element,
    detrend_3d_jax
)



class BaseDeformationDataset(ABC):
    def __init__(self, n_samples, gamma_range, seed=None, noise=0.0, mat_model: BaseMaterialModel = None):
        self.n_samples = n_samples
        self.noise = noise
        self.gamma_range = gamma_range
        self.mat_model = mat_model
        self.seed = seed
    def solve_plane_stress(self, lambda1):
        """
        Solve for lambda2 and lambda3 such that sigma22 = sigma33 = 0.
        Works for anisotropic materials.
        """

        def residuals(lams):
            lambda2, lambda3 = lams

            F = jnp.diag(jnp.array([lambda1, lambda2, lambda3]))
            sigma = self.get_cauchy_stress(F[None, :, :])[0]  # (3,3)

            return jnp.array([
                sigma[1, 1],   # σ22 = 0
                sigma[2, 2]    # σ33 = 0
            ])

        # good initial guess = isochoric lateral stretches
        lam0 = jnp.array([
            1.0 / jnp.sqrt(lambda1),
            1.0 / jnp.sqrt(lambda1)
        ])

        sol =root(residuals, lam0, method="hybr")
        return sol.x  # returns (lambda2, lambda3)
    @abstractmethod
    def get_F(self):
        """Abstract method to generate deformation gradient F (3x3 tensors)."""
        pass

    def get_invariants(self, deformation_gradient):
        B_train = B_func(deformation_gradient)
        I1_train = I1_func(B_train)
        I2_train = I2_func(B_train)
        I3_train = I3_func(B_train)
        return jnp.stack([I1_train, I2_train, I3_train], axis=-1)

    def get_phi(self, deformation_gradient):
        Phi = self.mat_model.phi(deformation_gradient)
        return Phi

    def get_cauchy_stress(self, deformation_gradient):
        """
        Computes Cauchy stress σ = (1/J) * P * Fᵀ, where P = ∂Φ/∂F.
        Uses JAX autograd.
        """
        def Phi_single(F):
            F = F.reshape(3, 3)
            return self.get_phi(F[None, :, :])[0]

        # Compute Piola stress P = dΦ/dF for each sample
        Phi_grad_fn = jax.vmap(jax.grad(Phi_single))
        Piola = Phi_grad_fn(deformation_gradient)
        # Piola = self.mat_model.P(deformation_gradient)

        J = J_func(deformation_gradient)
        sigma = jnp.einsum('nij,nkj->nik', Piola, deformation_gradient) / J[:, None, None]

        if self.noise > 0:
            key = jax.random.PRNGKey(self.seed if self.seed is not None else 0)
            noise = jax.random.normal(key, sigma.shape) * self.noise * sigma
            sigma = sigma + noise

        return sigma
    def get_piola_stress(self, deformation_gradient) :
        sigma = self.get_cauchy_stress(deformation_gradient)
        J = J_func(deformation_gradient)
        piola = J[:, None, None] * sigma @ jnp.linalg.inv(jnp.swapaxes(deformation_gradient, -2, -1))
        return piola
    def get_coeffs(self, deformation_gradient):
        B_train = B_func(deformation_gradient)
        sigma = self.get_cauchy_stress(deformation_gradient)

        # Eigenvalues
        B_eig_val = jnp.real(jnp.linalg.eigvalsh(B_train))
        sigma_eig_val = jnp.real(jnp.linalg.eigvalsh(sigma))

        coeffs, _ = solve_for_coefficients_batched(B_eig_val, sigma_eig_val)
        return coeffs


# ----------------------------
#   DEFORMATION MODES
# ----------------------------

class RandomFGenerator(BaseDeformationDataset):
    def get_F(self):
        F_np = generate_random_F_plane_stress(
            self.n_samples, self.gamma_range, self.seed
        )
        return jnp.array(F_np)


class UniaxialGenerator(BaseDeformationDataset):
    # def get_F(self):
    #     gamma = jnp.linspace(self.gamma_range[0], self.gamma_range[1], self.n_samples)
    #     F = jnp.zeros((self.n_samples, 3, 3))
    #     F = F.at[:, 0, 0].set(gamma)
    #     F = F.at[:, 1, 1].set(1.0)
    #     F = F.at[:, 2, 2].set(1.0)
    #     return F
    def get_F(self):
        gamma = jnp.linspace(self.gamma_range[0],
                             self.gamma_range[1],
                             self.n_samples)

        F_out = jnp.zeros((self.n_samples, 3, 3))

        for i, lam1 in enumerate(gamma):
            lam2, lam3 = self.solve_plane_stress(lam1)

            F = jnp.diag(jnp.array([lam1, lam2, lam3]))
            F_out = F_out.at[i].set(F)

        return F_out


class BiaxialGenerator(BaseDeformationDataset):
    def get_F(self):
        gamma = jnp.linspace(self.gamma_range[0], self.gamma_range[1], self.n_samples)
        F = jnp.zeros((self.n_samples, 3, 3))
        F = F.at[:, 0, 0].set(gamma)
        F = F.at[:, 1, 1].set(gamma)
        F = F.at[:, 2, 2].set(1.0)
        return F


class PureShearGenerator(BaseDeformationDataset):
    def get_F(self):
        gamma = jnp.linspace(self.gamma_range[0], self.gamma_range[1], self.n_samples)
        F = jnp.zeros((self.n_samples, 3, 3))
        F = F.at[:, 0, 0].set(gamma)
        F = F.at[:, 1, 1].set(1.0 / gamma)
        F = F.at[:, 2, 2].set(1.0)
        return F


class SimpleShearGenerator(BaseDeformationDataset):
    def get_F(self):
        gamma = jnp.linspace(self.gamma_range[0], self.gamma_range[1], self.n_samples)
        F = jnp.zeros((self.n_samples, 3, 3))
        F = F.at[:, 0, 0].set(1.0)
        F = F.at[:, 1, 1].set(1.0)
        F = F.at[:, 2, 2].set(1.0)
        F = F.at[:, 0, 1].set(gamma)
        F = F.at[:, 1, 0].set(0.0)
        return F
    
class BenchmarkDataset:
    def __init__(self, data_dir: os.PathLike, noise: str, mat_model: str):
        self.data_dir = data_dir
        self.noise = noise
        self.mat_model = mat_model
        self.mat_model_path = os.path.join(data_dir, noise, mat_model)
        self.loadsteps = os.listdir(self.mat_model_path)

    def __len__(self):

        return len(self.loadsteps)
    
    def __getitem__(self, loadstep):
        files_path = os.path.join(self.mat_model_path, str(loadstep))
        files = os.listdir(files_path)
        data = dict()
        for f in files :
            if f.endswith(".csv") :
                data[f"{f.split(".")[0]}"] = pd.read_csv(files_path + "/" + f)
        output_nodes = data["output_nodes"]
        output_elements = data["output_elements"]
        output_reactions = data["output_reactions"]
        output_integrator = data["output_integrator"]
        if 'ux_orig' in output_nodes.columns and 'uy_orig' in output_nodes.columns:
            output_nodes.ux[output_nodes.bcx!=0] = output_nodes.ux_orig[output_nodes.bcx!=0]
            output_nodes.uy[output_nodes.bcy!=0] = output_nodes.uy_orig[output_nodes.bcy!=0]
        mesh_pos = output_nodes[["x", "y"]].to_numpy()
        # u = output_nodes[["ux_orig", "uy_orig"]].to_numpy()

        u = output_nodes[["ux", "uy"]].to_numpy()
        bc = output_nodes[["bcx", "bcy"]].to_numpy()

        cells = output_elements[["node1", "node2", "node3"]].to_numpy()
        P2 = output_elements[["Pxx", "Pxy", "Pyx", "Pyy"]].to_numpy()
        F2 = output_elements[["Fxx", "Fxy", "Fyx", "Fyy"]].to_numpy()

        coords_elems = mesh_pos[cells]
        disp_elems   = u[cells]

        # Vectorize the function
        f = jax.vmap(lambda ce, de: deformation_gradient_element(ce, de))(coords_elems, disp_elems)


                # Reshape to (n, 2, 2)
        # --- 1. reshape ---
        P2 = P2.reshape(-1, 2, 2)
        F2 = F2.reshape(-1, 2, 2)

        # --- 2. create output arrays ---
        P = jnp.zeros((P2.shape[0], 3, 3))
        F = jnp.tile(jnp.eye(3), (P2.shape[0], 1, 1))

        # --- 3. JAX-style assignment ---
        P = P.at[:, :2, :2].set(P2)
        F = F.at[:, :2, :2].set(F2)
        mm = get_material(self.mat_model.lower())
        P_from_mm = mm.P(F)
        J = J_func(F)

        sigma = 1/J[:, None, None] * P_from_mm @ jnp.swapaxes(F, -2, -1)
        B_train = B_func(F)
        I1_train = I1_func(B_train)
        I2_train = I2_func(B_train)
        I3_train = I3_func(B_train)
        invariants = jnp.stack([I1_train, I2_train, I3_train], axis=-1)

        # Eigenvalues
        B_eig_val = jnp.real(jnp.linalg.eigvalsh(B_train))
        sigma_eig_val = jnp.real(jnp.linalg.eigvalsh(sigma))

        coeffs, _ = solve_for_coefficients_batched(B_eig_val, sigma_eig_val)

        reaction_forces = output_reactions["forces"] #right, left, top, bottom 
        data = dict(F = F, P = P_from_mm, sigma = sigma, coeffs = coeffs, invariants = invariants, 
                    cells = cells, coords_elems = coords_elems, disp_elems = disp_elems, bc = bc, reaction_forces = reaction_forces)
        return data

class TractionDataset :
    def __init__(self, data_dir: os.PathLike = "/home/mmdiscovery/shared/dataset/isihara_fix"):
        self.data_dir = data_dir
        self.files = os.listdir(self.data_dir)
    def __len__(self) :
        return len(self.files)
    def __getitem__(self, idx) :
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        return data

        
# Example usage
if __name__ == "__main__":
    dataset = BenchmarkDataset("dataset/benchmarks", "noise=low", "Isihara")
    loadsteps = [10, 50, 80]
    check = dataset[10]
    data = TestSpecimen("dataset/benchmarks/test-specimen", "Isihara-GT")
    check = data[10]
    print('')
    # dataset = UniaxialGenerator(n_samples=10, gamma_range=(0.5, 2.0), mat_model="MooneyRivlin")
    # F = dataset.get_F()
    # invariants = dataset.get_invariants(F)
    # sigma = dataset.get_cauchy_stress(F)
    # coeffs = dataset.get_coeffs(F)

    # print("F shape:", F.shape)
    # print("Invariants shape:", invariants.shape)
    # print("Cauchy stress shape:", sigma.shape)
    # print("Coefficients shape:", coeffs.shape)

