import torch
def NH_model(deformation_gradient: torch.Tensor) :
    jacobian = torch.det(deformation_gradient)
    invariant_1 = jacobian** (-2/3) * torch.trace(deformation_gradient)
    free_energy = 0.5 * (invariant_1 - 3) + 1.5 * (jacobian - 1) ** 2
    return free_energy