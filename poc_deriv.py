from core.StrainEnergyGP import StrainEnergyGP
from core.utils import * 
import matplotlib.pyplot as plt


if __name__ == "__main__" :
    # Example: batch of N 3x3 matrices
    n_data = 100
    Phi_noise_percent = 0.03
    gamma = torch.linspace(0.0, 10.0, n_data)
    F_ut =  torch.zeros((n_data, 3, 3))

    # Uniaxial Case
    F_ut[:, 0, 0] = 1 + gamma
    F_ut[:, 1, 1] = 1
    F_ut[:, 2, 2] = 1

    F_ut.requires_grad=True

    B = B_func(F_ut)
    I1 = I1_func(B)
    I2 = I2_func(B)
    I3 = I3_func(B)

    Phi = MooneyRivlinPhi(I1, I2, I3)
    noise_level =  Phi_noise_percent * Phi.mean()
    noise = torch.normal(torch.zeros_like(Phi), torch.ones_like(Phi)) * noise_level
    Phi_noised = Phi + noise

    Piola_noised = torch.autograd.grad(Phi_noised.clone().sum(), F_ut, retain_graph = True)[0]  # same shape as f [3,3]
    tau_noised = Piola_noised @ F_ut.transpose(-2, -1)
    sigma_noised = tau_noised / J_func(F_ut)[:, None, None]


    fig, axs = plt.subplots(1, 4, figsize=(12, 5))  # 1 row, 2 columns

    # First subplot: Phi vs gamma
    # axs[0].plot(I1.detach().numpy(), Phi.detach().numpy(), label=r'$\Phi$')
    # axs[0].plot(I2.detach().numpy(), Phi.detach().numpy(), label=r'$\Phi$')
    # axs[0].plot(I3.detach().numpy(), Phi.detach().numpy(), label=r'$\Phi$')
    axs[0].plot(gamma.numpy(), Phi.detach().numpy(), color="b", label=r'$\Phi$')
    axs[0].scatter(gamma.numpy(), Phi_noised.detach().numpy(), color="k", marker="x", label=r'Noised $\Phi$')
    axs[0].set_xlabel(r'$\gamma$')
    axs[0].set_ylabel(r'$\Phi$')
    axs[0].set_title(r'$\Phi$ Mooney Rivlin')
    axs[0].grid(True)
    axs[0].legend()

    # Second subplot: tau_noised components vs gamma
    axs[1].scatter(gamma.numpy(), tau_noised[:, 0, 0].detach().numpy(), color="r", marker="x", label=r'$\tau_{00}$')
    axs[1].scatter(gamma.numpy(), tau_noised[:, 1, 1].detach().numpy(), color="g", marker="+", label=r'$\tau_{11}$')
    axs[1].scatter(gamma.numpy(), tau_noised[:, 2, 2].detach().numpy(), color="b", marker="o", label=r'$\tau_{22}$')
    axs[1].set_xlabel(r'$\gamma$')
    axs[1].set_ylabel(r'$\tau$ Mooney Rivlin')
    axs[1].set_title(r'$\tau$ Mooney Rivlin')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].scatter(gamma.numpy(), Piola_noised[:, 0, 0].detach().numpy(), color="r", marker="x", label=r'$P_{00}$')
    axs[2].scatter(gamma.numpy(), Piola_noised[:, 1, 1].detach().numpy(), color="g", marker="+", label=r'$P_{11}$')
    axs[2].scatter(gamma.numpy(), Piola_noised[:, 2, 2].detach().numpy(), color="b", marker="o", label=r'$P_{22}$')
    axs[2].set_xlabel(r'$\gamma$')
    axs[2].set_ylabel(r'$P$ Mooney Rivlin')
    axs[2].set_title(r'$P$ Mooney Rivlin')
    axs[2].grid(True)
    axs[2].legend()

    axs[3].scatter(gamma.numpy(), sigma_noised[:, 0, 0].detach().numpy(), color="r", marker="x", label=r'$\sigma_{00}$')
    axs[3].scatter(gamma.numpy(), sigma_noised[:, 1, 1].detach().numpy(), color="g", marker="+", label=r'$\sigma_{11}$')
    axs[3].scatter(gamma.numpy(), sigma_noised[:, 2, 2].detach().numpy(), color="b", marker="o", label=r'$\sigma_{22}$')
    axs[3].set_xlabel(r'$\gamma$')
    axs[3].set_ylabel(r'$\sigma$ Mooney Rivlin')
    axs[3].set_title(r'$\sigma$ Mooney Rivlin')
    axs[3].grid(True)
    axs[3].legend()

    plt.tight_layout()
    plt.show()


