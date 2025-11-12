import json
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .utils import * 
class BenchmarkDataset(Dataset):
    def __init__(self, data_dir: os.PathLike, noise: str, mat_model: str):

        super(BenchmarkDataset, self).__init__()
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

        mesh_pos = output_nodes[["x", "y"]].to_numpy()
        # u = output_nodes[["ux_orig", "uy_orig"]].to_numpy()

        u = output_nodes[["ux", "uy"]].to_numpy()
        bc = torch.tensor(output_nodes[["bcx", "bcy"]].to_numpy())

        cells = output_elements[["node1", "node2", "node3"]].to_numpy()
        P2 = torch.tensor(output_elements[["Pxx", "Pxy", "Pyx", "Pyy"]].to_numpy(), dtype=torch.float32)
        F2 = torch.tensor(output_elements[["Fxx", "Fxy", "Fyx", "Fyy"]].to_numpy(), dtype=torch.float32)

        # Reshape to (n, 2, 2)
        P2 = P2.view(-1, 2, 2)
        F2 = F2.view(-1, 2, 2)

        # --- 2. Embed into 3×3 tensors (pad zeros) ---
        P = torch.zeros((P2.shape[0], 3, 3), dtype=torch.float32)
        F = torch.eye(3, dtype=torch.float32).repeat(P2.shape[0], 1, 1)

        P[:, :2, :2] = P2
        F[:, :2, :2] = F2

        # --- 3. Compute Cauchy stress ---
        # σ = (1/J) * P * Fᵀ
        J = torch.det(F)
        sigma = torch.bmm(P, F.transpose(1, 2)) / J[:, None, None]
        # sigma_stress = piola_stress
        gradNa = []
        for i in range(3):
            gradNa.append(output_integrator[[f"gradNa_node{i+1}_x", f"gradNa_node{i+1}_y"]].to_numpy())

        gradNa = torch.tensor(gradNa).permute(1, 0, 2)
        qpWeight =  torch.tensor(output_integrator[["qpWeight"]].to_numpy())
        voigtMap = [[0,1],[2,3]]
        u_a = torch.tensor(u[cells])
        X_a = torch.tensor(mesh_pos[cells])
        # deformation_gradient = torch.zeros(cells.shape[0], 4, dtype = torch.float64) #noised deformation
        # for a in range(3):
        #     for i in range(2):
        #         for j in range(2):
        #             deformation_gradient[:,voigtMap[i][j]] += u_a[:, a, i] * gradNa[:, a, j]
        # deformation_gradient[:,0] += 1.0
        # deformation_gradient[:,3] += 1.0
        # check = torch.max(f - deformation_gradient)
        data = dict(F = F, u_a = u_a, qpWeight = qpWeight, gradNa = gradNa, cauchy_stress = sigma, cells = cells, X_a = X_a, bc = bc)
        return data
class RandomKnownModelGenerator() :
    def __init__(self, n_samples, gamma_range, noise, mat_model):

        self.n_samples = n_samples
        self.noise = noise
        self.gamma_range = gamma_range
        self.mat_model = mat_model
    
    def get_F(self) :
        return torch.tensor(generate_random_F_2D(self.n_samples, self.gamma_range))
    
    def get_invariants(self, deformation_gradient) :
        B_train = B_func(deformation_gradient)

        I1_train = I1_func(B_train)
        I2_train = I2_func(B_train)
        I3_train = I3_func(B_train)
        return torch.stack([I1_train, I2_train, I3_train]).T
    
    def get_phi(self, deformation_gradient) :
        B_train = B_func(deformation_gradient)

        I1_train = I1_func(B_train)
        I2_train = I2_func(B_train)
        I3_train = I3_func(B_train)

        # --- 4. Calculate Stress/Energy (Targets) ---
        # Calculate the Strain Energy Potential (Phi)
        Phi = MooneyRivlinPhi(I1_train, I2_train, I3_train) # MN for now

        return Phi
    
    def get_cauchy_stress(self, deformation_gradient) :
        deformation_gradient.requires_grad = True
        Phi = self.get_phi(deformation_gradient)

        Piola = torch.autograd.grad(
            Phi.clone().sum(), 
            deformation_gradient, 
            retain_graph=True
        )[0] 

        sigma = Piola @ deformation_gradient.transpose(-2, -1) / J_func(deformation_gradient)[:, None, None]

        # --- 5. Add Noise and Prepare GP Targets ---
        noise_percentage = self.noise
        noise_level = noise_percentage * sigma
        noise = noise_level * torch.normal(torch.zeros_like(sigma), torch.ones_like(sigma))
        sigma_noised = sigma + noise
        return sigma_noised.detach()
    
    def get_coeffs(self, deformation_gradient) :

        B_train = B_func(deformation_gradient)
        sigma = self.get_cauchy_stress(deformation_gradient)

        sigma_eig_val, _ = torch.linalg.eig(sigma)
        B_eig_val, _ = torch.linalg.eig(B_train)

        coeffs, _ = solve_for_coefficients_batched(B_eig_val.to(dtype = B_train.dtype), sigma_eig_val.to(dtype = sigma.dtype))

        return coeffs.detach()


if __name__ == "__main__" :
    data_dir = "/home/mmdiscovery/shared/dataset/benchmarks/"
    noise = "noise=low"
    mat_model = "NeoHookean"
    dataset = BenchmarkDataset(data_dir, noise, mat_model)
    data = dataset[10]