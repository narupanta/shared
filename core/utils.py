import torch
import gpytorch
import numpy as np
def J_func(f):
    return torch.linalg.det(f)  # shape [N]

def B_func(f):
    return f @ f.transpose(-2,-1)  # [N,d,d]

def I1_func(B):
    return torch.einsum('...ii->...', B)  # trace along last two dims

def I2_func(B):
    trB = torch.einsum('...ii->...', B)
    trBB = torch.einsum('...ii->...', B @ B)
    return 0.5 * (trB**2 - trBB)
def I3_func(B):
    return torch.linalg.det(B)  # shape [N]

def MooneyRivlinPhi(I1, I2, I3):
    # c1, c2, c3 = 1.0, 0.001, 1000
    c1, c2, c3 = 0.162, 0.0059, 10
    term1 = c1 * (I3**(-0.5) * I1 - 3)
    term2 = c2 * (I3**(-2/3) * I2 - 3)
    term3 = c3 * (I3**0.5 - 1)**2
    return term1 + term2 + term3  # [N]
def NeoHookeanPhi(f):
    c1, c2 = 0.5, 1.5
    term1 = c1 * (torch.sqrt(I3_func(f))**(-2/3) * I1_func(f) - 3)
    term2 = c2 * (torch.sqrt(I3_func(f)) - 1) ** 2
    return term1 + term2

def solve_for_coefficients_batched(lambda_B, lambda_sigma):
    """
    Batched version of solve_for_coefficients.

    Args:
        lambda_B: Tensor of shape (batch_size, 3)
        lambda_sigma: Tensor of shape (batch_size, 3)

    Returns:
        coefficients_c: Tensor of shape (batch_size, 3)
        V: Vandermonde matrices of shape (batch_size, 3, 3)
    """

    # --- 1. Validate input shapes ---
    if lambda_B.ndim != 2 or lambda_B.shape[1] != 3:
        raise ValueError("lambda_B must have shape (batch_size, 3)")
    if lambda_sigma.ndim != 2 or lambda_sigma.shape[1] != 3:
        raise ValueError("lambda_sigma must have shape (batch_size, 3)")
    if lambda_B.shape[0] != lambda_sigma.shape[0]:
        raise ValueError("Batch sizes of lambda_B and lambda_sigma must match.")

    # --- 2. Construct Vandermonde matrices ---
    # V[i] = [[1, λ1, λ1²],
    #         [1, λ2, λ2²],
    #         [1, λ3, λ3²]]
    col1 = torch.ones_like(lambda_B)
    col2 = lambda_B
    col3 = lambda_B**2

    V = torch.stack((col1, col2, col3), dim=-1)  # (batch_size, 3, 3)

    # --- 3. Compute pseudoinverses for each batch ---
    # torch.linalg.pinv supports batched input
    V_pinv = torch.linalg.pinv(V)  # (batch_size, 3, 3)

    # --- 4. Solve for c ---
    # (batch_size, 3, 3) @ (batch_size, 3, 1) -> (batch_size, 3, 1)
    coefficients_c = torch.bmm(V_pinv, lambda_sigma.unsqueeze(-1)).squeeze(-1)

    return coefficients_c, V


def generate_random_F_2D(n_samples, lambda_range=(0.5, 2.5)):
    """
    Generates n_samples of 3x3 Deformation Gradient (F) tensors 
    constrained to 2D Plane Strain (F33=1, F_i3=0, F_3i=0 for i=1,2).
    
    The top-left 2x2 submatrix F_2D is generated randomly using polar decomposition
    and a random in-plane rotation.
    """
    F_list = []
    low, high = lambda_range
    
    # Calculate the number of shear samples to add
    n_shear = n_samples // 10
    
    for i in range(n_samples):
        # 1. Random 2x2 Stretch Matrix (V_2D - Diagonal)
        # Sample two principal stretches for the plane
        lambdas_2D = np.random.rand(2) * (high - low) + low
        V_2D = np.diag(lambdas_2D)
        
        # 2. Random 2x2 Rotation Matrix (R_2D - Orthogonal)
        # Random angle for in-plane rotation
        theta = np.random.rand() * 2 * np.pi
        R_2D = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # 3. 2x2 Deformation Gradient F_2D = R_2D @ V_2D
        F_2D_sub = R_2D @ V_2D
        
        # 4. Construct the 3x3 Plane Strain F
        F_sample = np.eye(3)
        F_sample[:2, :2] = F_2D_sub
        
        F_list.append(F_sample[None, :, :])

    # Add a set of Pure Shear states (Simple Shear in the 2D plane)
    gamma_shear = np.random.rand(n_shear) * (high - low) + low
    F_shear = np.tile(np.eye(3), (n_shear, 1, 1))
    
    # F_12 component for simple shear
    F_shear[:, 0, 1] = gamma_shear
    F_list.append(F_shear)
    
    return np.concatenate(F_list, axis=0)