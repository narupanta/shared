from core.utils import *
import os
from core.datasetclass import *
from core.model import TensorBasisGPModel, SVTBGPModel
import jax
import jax.numpy as jnp
import numpy as np
# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
from core.material_models import get_material
from sklearn.preprocessing import StandardScaler, MinMaxScaler
config.update("jax_enable_x64", True)

import datetime
# with install_import_hook("gpjax", "beartype.beartype"):
#     import gpjax as gpx
import gpjax as gpx

key = jr.key(123)


cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
def plot_R2_PK(pk1_pred, pk1_gt, filename="pk_plot.png"):
    """
    pk1_pred, pk1_gt: arrays of shape (N, 2, 2) or (N, 4)
    Saves a PNG plot comparing prediction vs ground truth.
    Adds a 45° reference line (y = x).
    """
    # Flatten to (N, 4)
    pred = pk1_pred.reshape(len(pk1_pred), -1)
    gt   = pk1_gt.reshape(len(pk1_gt), -1)

    # R² function
    def r2(y_pred, y_true):
        ss_res = jnp.sum((y_true - y_pred)**2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true))**2)
        return 1 - ss_res / ss_tot

    r2_vals = [float(r2(pred[:, i], gt[:, i])) for i in range(4)]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.reshape(-1)

    component_labels = ["P11", "P12", "P21", "P22"]

    for i in range(4):
        ax = axes[i]

        # Scatter plot
        ax.scatter(gt[:, i], pred[:, i])

        # 45° reference line
        min_val = min(gt[:, i].min(), pred[:, i].min())
        max_val = max(gt[:, i].max(), pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], color = "k", linestyle = "-")

        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{component_labels[i]}  (R² = {r2_vals[i]:.3f})")

    fig.suptitle("First Piola–Kirchhoff Stress: Prediction vs Ground Truth")
    fig.tight_layout()

    # Save as PNG
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return r2_vals

if __name__ == "__main__" :
    base_save_path = "saved_model"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)

    # Subfolder with datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    save_path = os.path.join(base_save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print("Saving everything to:", save_path)
    # dataset = TestSpecimen("dataset/benchmarks/test-specimen", "Isihara-GT")
    dataset = BenchmarkDataset("dataset/benchmarks", "noise=low", "Isihara")
    loadsteps = dataset.loadsteps[:]
    F_list = []
    invariants_list = []
    sigma_list = []
    coeffs_list = []
    piola_list = []
    for i in loadsteps :
        data = dataset[i]
        f = data["F"]
        invariants = data["invariants"]
        sigma = data["sigma"]
        coeffs = data["coeffs"]
        piola = data["P"]
        F_list.append(f)
        invariants_list.append(invariants)
        coeffs_list.append(coeffs)
        sigma_list.append(sigma)
        piola_list.append(piola)

    F_train = jnp.concat(F_list)
    invariants_train = jnp.concat(invariants_list)
    sigma_train = jnp.concat(sigma_list)
    coeffs_train = jnp.concat(coeffs_list)
    piola_train = jnp.concat(piola_list)
    key = jax.random.PRNGKey(0)

    N = invariants_train.shape[0]
    perm = jax.random.permutation(key, N)
    f_shuffled = F_train[perm]
    invariants_shuffled = invariants_train[perm]
    coeffs_shuffled = coeffs_train[perm]
    piola_shuffled = piola_train[perm]

    split_idx = int(0.8 * N)

    inv_train = invariants_shuffled[:split_idx]
    coeff_train = coeffs_shuffled[:split_idx]
    f_test = f_shuffled[split_idx:]
    inv_test = invariants_shuffled[split_idx:]
    coeff_test = coeffs_shuffled[split_idx:]
    piola_test = piola_shuffled[split_idx:]
    model = SVTBGPModel(inv_train, coeff_train, 100)
    opt_posteriors, _ = model.optimization(10000)
    model.save_model(save_path)
    model_eval = model
    model_eval.load_model(save_path)


    pk1_pred = model_eval.predict_piola_stress(f_test)[0]
    pred_coeffs = model_eval.predict_coeffs(inv_test)[0]
    plot_R2_PK(pk1_pred[:, :2, :2], piola_test[:, :2, :2], filename=save_path + f"/pk_plot_test.png")

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    # pred: (N, 3)
    # gt:   (N, 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, ax in enumerate(axes):
        # Scatter plot
        ax.scatter(coeff_test[:, i], pred_coeffs[:, i], alpha=0.7)

        # R² score
        r2 = r2_score(coeff_test[:, i], pred_coeffs[:, i])
        min_val = min(coeff_test[:, i].min(), pred_coeffs[:, i].min())
        max_val = max(coeff_test[:, i].max(), pred_coeffs[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')
        ax.set_xlabel(f"c{i+1} ground truth")
        ax.set_ylabel(f"c{i+1} prediction")
        ax.set_title(f"c{i+1} (R² = {r2:.3f})")

    plt.tight_layout()
    fig.savefig(save_path + f"/coeffs_plot_test.png", dpi=300)

    # eval
    # for h in [10, 20, 30, 40, 50, 60, 70, 80]:
    #     data = dataset[h]
    #     skip = 1
    #     f = data["F"][::skip]
    #     invariants = data["invariants"]
    #     sigma = data["sigma"]
    #     coeffs = data["coeffs"]
    #     piola = data["P"]
    #     coeffs_means, coeffs_stds = model.predict_coeffs(invariants)
    #     pk1_pred = model_eval.predict_piola_stress(f)[0]
    #     plot_R2_PK(pk1_pred[:, :2, :2], piola[:, :2, :2], filename=save_path + f"/pk_plot{h}.png")

    # with test specimens
    # test_dataset = TestSpecimen("dataset/benchmarks/test-specimen", "Isihara-GT")
    # # test_dataset = BenchmarkDataset("dataset/benchmarks", "noise=low", "Isihara")
    # for step in test_dataset.loadsteps :
    #     skip = 2
    #     data = test_dataset[step]
    #     f = data["F"][::skip]
    #     invariants = data["invariants"][::skip]
    #     sigma = data["sigma"][::skip]
    #     coeffs = data["coeffs"][::skip]
    #     piola = data["P"][::skip]
    #     # coeffs_means, coeffs_stds = model.predict_coeffs(invariants)
    #     pk1_pred = model_eval.predict_piola_stress(f)[0]
    #     plot_R2_PK(pk1_pred[:, :2, :2], piola[:, :2, :2], filename=save_path + f"/pk_plot_test_{step}.png")
        # sigma_mean, sigma_std = model_eval.predict_cauchy_stress(f)
#
        # plot predicted coeffs for each deformation modes subplots #gamma_range vs sigma sigma11, sigma22, sigma33, sigma12, sigma13, sigma23
        # and save to the save path
        # gamma_range = gen.gamma_range
        # n_points = gen.n_samples

        # gamma = jnp.linspace(gamma_range[0], gamma_range[1], n_points)
        # fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        # stress_names = ["11", "22", "33", "12", "13", "23"]

        # comps = [
        #     (0, 0), (1, 1), (2, 2),
        #     (0, 1), (0, 2), (1, 2)
        # ]

        # for ax, (a, b), name in zip(axs.flatten(), comps, stress_names):
        #     mean = sigma_mean[:, a, b]
        #     std = sigma_std[:, a, b]
        #     ax.scatter(gamma, sigma[:, a, b], marker = "x", color = "k")
        #     ax.plot(gamma, mean)
        #     # ax.fill_between(gamma, mean - 2 * std, mean + 2 * std, alpha=0.3)
        #     ax.set_xlabel("gamma")
        #     ax.set_ylabel(f"sigma{name}")

        # fig.suptitle(f"Cauchy Stress Prediction - {h}")
        # fig.tight_layout()
        # fig.savefig(f"{save_path}/stress_{h}.png")
        # plt.close()

    # plot stress wrt to gamma Uniaxial, Biaxial, Pureshear

    gamma_range = (0.5, 2)
    sample = 200
    ut = UniaxialGenerator(sample, gamma_range, None, 0, get_material("isihara"))
    f_ut = ut.get_F()
    piola_ut_gt = ut.get_piola_stress(f_ut)
    piola_ut_pred = model_eval.predict_piola_stress(f_ut)[0]



    # gamma axis
    gamma = np.linspace(gamma_range[0], gamma_range[1], sample)

    # piola_ut_gt: shape (N, 3, 3)
    # piola_ut_pred: shape (N, 3, 3)

    # Extract components
    P11_gt = piola_ut_gt[:, 0, 0]
    P12_gt = piola_ut_gt[:, 0, 1]
    P21_gt = piola_ut_gt[:, 1, 0]
    P22_gt = piola_ut_gt[:, 1, 1]

    P11_pred = piola_ut_pred[:, 0, 0]
    P12_pred = piola_ut_pred[:, 0, 1]
    P21_pred = piola_ut_pred[:, 1, 0]
    P22_pred = piola_ut_pred[:, 1, 1]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    components = [
        ("P11", P11_gt, P11_pred),
        ("P12", P12_gt, P12_pred),
        ("P21", P21_gt, P21_pred),
        ("P22", P22_gt, P22_pred),
    ]

    for ax, (title, gt, pred) in zip(axes.ravel(), components):
        ax.plot(gamma, gt, label="Ground Truth")
        ax.plot(gamma, pred, label="Predicted")
        ax.set_title(title)
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Piola")
        ax.legend()

    plt.tight_layout()
    fig.savefig(f"{save_path}/stress_ut.png")



