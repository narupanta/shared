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

    model_date_time = "20251126T143045"
    model_path = f"saved_model/{model_date_time}"
    gp_model = SVTBGPModel()
    gp_model.load_model(model_path)
    test_dataset = TestSpecimen("dataset/benchmarks/test-specimen", "Isihara-GT")
    for step in test_dataset.loadsteps :
        print(step)
        data = test_dataset[step]
        f = data["F"]
        invariants = data["invariants"]
        sigma = data["sigma"]
        coeffs = data["coeffs"]
        piola = data["P"]
        coeffs_means, coeffs_stds = gp_model.predict_coeffs(invariants)
        pk1_pred = gp_model.predict_piola_stress(f)[0]
        plot_R2_PK(pk1_pred[:, :2, :2], piola[:, :2, :2], filename=model_path + f"/pk_plot_test_{step}.png")