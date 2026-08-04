import jax 
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP
from core.utils import transform_input_features
from core.dataclass import GPRawParams, GPParams, GPWeights
from core.material_models import get_material
from core.trainer import HyperelasticGPTrainer
from core.features import IsotropicFeatureExtractor
from core.utils import *
import datetime
import os
from tqdm import tqdm
from core.datasetclass import TractionDataset
from core.loss_function import total_stochastic_loss
from core.plotter import \
    plot_loss_analysis, \
    plot_parameters_hist, plot_inducing_points, plot_combined_validation, plot_training_r2
# helper: per-element edge-based neumann traction contribution
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--material_model_name', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.0001)
    parser.add_argument('--load_noise', type=float, default=0.01)
    parser.add_argument('--target_load_true_top', type=float, default=8.0)
    parser.add_argument('--asym_factor', type=float, default=0.95)

    # Training Config
    parser.add_argument('--number_of_mci_sampling', type=int, default=3)
    parser.add_argument('--n_ip', type=int, default=5)
    parser.add_argument('--beta', type=float, default=50.0)
    
    # Booleans (using 0/1 as integers is often safer in shell scripts)
    parser.add_argument('--is_fixed_reaction_force_noise', type=int, default=1)

    # Handling the List [1, 5, 9] to cover the 10 steps range
    parser.add_argument('--train_load_steps_indices', type=int, nargs='+', default=[1, 5, 9])
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default="", help="Name of the extraction/extracted_models folder to resume from")

    return parser.parse_args()

def sigma_fix_to_log_sigma_fix(sigma_fix) :
    return jnp.log(jnp.maximum(sigma_fix, 1e-3))



def freeze_reaction_force_noise(grads) :
    return grads._replace(
        log_sigma_fix_x=jnp.zeros_like(grads.log_sigma_fix_x),
        log_sigma_fix_y=jnp.zeros_like(grads.log_sigma_fix_y))



if __name__ == "__main__" :
    base_save_path = "extraction/extracted_models"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)
    # training_mode = "stochastic"
    args = parse_args()

    # Now use args.variable_name instead of hardcoded values
    material_model_name = args.material_model_name

    disp_noise = args.disp_noise
    load_noise = args.load_noise
    target_load_true_top = args.target_load_true_top
    asym_factor = args.asym_factor
    number_of_mci_sampling = args.number_of_mci_sampling
    train_load_steps_indices = args.train_load_steps_indices
    n_ip = args.n_ip
    beta = args.beta
    is_fixed_reaction_force_noise = args.is_fixed_reaction_force_noise

    n_iterations = args.n_iterations
    learning_rate = args.learning_rate

    # Subfolder with datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    training_config_str = f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{n_ip}_{beta}_{is_fixed_reaction_force_noise}"
    save_path = os.path.join(base_save_path, f"{timestamp}_{training_config_str}")
    os.makedirs(save_path, exist_ok=True)

    # load precomputed dataset
    from core.datasetclass import DatasetFactory
    data_dir = "dataset/preprocessed/syn_f" if os.path.exists("dataset/preprocessed/syn_f") else "dataset/precomputed_vfm" 
    prep_dataset_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}.npz")
    
    dataset = DatasetFactory.create("dataset/precomputed_vfm", data_path=prep_dataset_path)
    prep_data = dataset.get_data()
    f2x2 = prep_data["F"][train_load_steps_indices] 

    # Data use in VFM
    f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2)
    f_neu_nodes = prep_data["f_neu"][train_load_steps_indices] 
    node_type = np.asarray(prep_data["node_type"])
    dNdX = prep_data["dNdX"]
    dA = prep_data["dA"]
    cells = prep_data["cells"]
    load_noise_std = prep_data["load_noise_std"]
    load_noise_std_steps = prep_data["load_noise_std_steps"][train_load_steps_indices] 

    true_mat_model = get_material(material_model_name)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)

    extractor = IsotropicFeatureExtractor()
    dev, vol = jax.vmap(jax.vmap(extractor.extract))(f3x3)
    I_all = jnp.concatenate([dev, vol], axis=-1)
    # get all data inside prep_data
    dev_flat =  dev.reshape(-1, dev.shape[-1]) 
    vol_flat = vol.reshape(-1, vol.shape[-1])
    
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
        resume_dir = os.path.join(base_save_path, args.resume_from)
        I_z = jnp.load(os.path.join(resume_dir, "I_z.npy"))
        dev_z = I_z[:, :2]
        vol_z = I_z[:, 2:]
    else:
        dev_z = farthest_point_sampling_with_fixed_point(dev_flat, n_ip, jnp.array([3.0, 3.0]))
        vol_z = farthest_point_sampling_with_fixed_point(vol_flat, n_ip, jnp.array([1.0]))
        I_z = jnp.concat([dev_z, vol_z], axis = -1)
        
    plot_inducing_points(dev_z, vol_z, dev_flat, vol_flat, save_path)

    # Setup random key
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    if args.resume_from:
        resume_dir = os.path.join(base_save_path, args.resume_from)
        best_params_dict = np.load(os.path.join(resume_dir, "best_params.npy"), allow_pickle=True).item()
        params = GPRawParams(**best_params_dict)
    else:
        if is_fixed_reaction_force_noise:
            params = GPRawParams(
                # Lengthscales and signal variances (Normal(0, 1))
                raw_dev_ls=jax.random.normal(k1, (2,)),
                raw_dev_sig=jax.random.normal(k1, ()),
                
                # Inducing point means and variances
                raw_dev_z =jax.random.normal(k2, (n_ip, 2)),
                raw_dev_u_mean=jax.random.normal(k2, (n_ip,)),
                raw_dev_u_var=jax.random.normal(k2, (n_ip,)),

                raw_vol_ls=jax.random.normal(k3, (1,)),
                raw_vol_sig=jax.random.normal(k3, ()),

                raw_vol_z =jax.random.normal(k4, (n_ip,1)),        
                raw_vol_u_mean=jax.random.normal(k4, (n_ip,)),
                raw_vol_u_var=jax.random.normal(k4, (n_ip,)),
                raw_vol_kappa=jnp.array(0.0),



                # Noise parameters (Fixed PDE residual noise to prevent uncertainty collapse)
                log_sigma_free_x=jnp.log(jnp.array(1.0)),
                log_sigma_free_y=jnp.log(jnp.array(1.0)),
                log_sigma_fix_x=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 0]),
                log_sigma_fix_y=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 1])
                )
        else :
            params = GPRawParams(
                # Lengthscales and signal variances (Normal(0, 1))
                raw_dev_ls=jax.random.normal(k1, (2,)),
                raw_dev_sig=jax.random.normal(k1, ()),
                
                # Inducing point means and variances
                raw_dev_z =jax.random.normal(k2, (n_ip, 2)),
                raw_dev_u_mean=jax.random.normal(k2, (n_ip,)),
                raw_dev_u_var=jax.random.normal(k2, (n_ip,)),

                raw_vol_ls=jax.random.normal(k3, (1,)),
                raw_vol_sig=jax.random.normal(k3, ()),

                raw_vol_z =jax.random.normal(k4, (n_ip,1)),        
                raw_vol_u_mean=jax.random.normal(k4, (n_ip,)),
                raw_vol_u_var=jax.random.normal(k4, (n_ip,)),
                raw_vol_kappa=jnp.array(0.0),



                # Noise parameters (Fixed PDE residual noise to prevent uncertainty collapse)
                log_sigma_free_x=jnp.log(jnp.array(1.0)),
                log_sigma_free_y=jnp.log(jnp.array(1.0)),
                log_sigma_fix_x=jax.random.normal(k3, (load_noise_std_steps.shape[0],)),
                log_sigma_fix_y=jax.random.normal(k4, (load_noise_std_steps.shape[0],))
            )
    
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    main_key = jr.PRNGKey(42)

    model = SparseHyperelasticityGP(params, I_z, min_dev, min_vol, max_dev, max_vol, beta = beta)




    loss_fn = lambda p, k: total_stochastic_loss(p, model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type, dNdX, dA, k, number_of_mci_sampling)

    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)
    
    trainer = HyperelasticGPTrainer(
        model=model,
        initial_params=params,
        loss_fn=loss_fn,
        opt_state=opt_state,
        optimizer=opt,
        save_path=save_path,
        true_mat_model=true_mat_model,
        I_z=I_z,
        I_all=I_all,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        freeze_fn=freeze_reaction_force_noise if is_fixed_reaction_force_noise else None
    )

    log_info_str = f"{train_load_steps_indices}, {material_model_name}"
    best_params = trainer.train(n_iterations=n_iterations, main_key=main_key, log_info_str=log_info_str)

    print("Generating Training Data R2 Plot for all load steps...")
    learned_gp = SparseHyperelasticityGP(best_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=beta)
    F_train_full_3x3 = jax.vmap(jax.vmap(fto3x3))(prep_data["F"])
    plot_training_r2(learned_gp, true_mat_model, F_train_full_3x3, save_path)

    print(f"{timestamp}_{training_config_str}")


