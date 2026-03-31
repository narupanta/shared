from typing import NamedTuple
import jax.numpy as jnp

class EnergyDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray

class StressDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray

class GPRawParams(NamedTuple):
    raw_dev_ls: jnp.ndarray
    raw_dev_sig: jnp.ndarray
    raw_dev_u_mean: jnp.ndarray
    raw_dev_u_var: jnp.ndarray
    raw_dev_z: jnp.ndarray


    raw_vol_ls: jnp.ndarray
    raw_vol_sig: jnp.ndarray
    raw_vol_u_mean: jnp.ndarray
    raw_vol_u_var: jnp.ndarray
    raw_vol_z: jnp.ndarray

    raw_c01: jnp.ndarray
    raw_c02: jnp.ndarray
    raw_c10: jnp.ndarray
    raw_c11: jnp.ndarray
    raw_c20: jnp.ndarray
    raw_k: jnp.ndarray
    raw_q: jnp.ndarray
    raw_s: jnp.ndarray


    log_sigma_free_x: jnp.ndarray
    log_sigma_free_y: jnp.ndarray
    log_sigma_fix_x: jnp.ndarray
    log_sigma_fix_y: jnp.ndarray

class RawMatParams(NamedTuple) :
    raw_c01: jnp.ndarray
    raw_c02: jnp.ndarray
    raw_c10: jnp.ndarray
    raw_c11: jnp.ndarray
    raw_c20: jnp.ndarray    
    raw_k: jnp.ndarray
    raw_q: jnp.ndarray
    raw_s: jnp.ndarray  
class MatParams(NamedTuple) :
    c01: jnp.ndarray
    c02: jnp.ndarray
    c10: jnp.ndarray
    c11: jnp.ndarray
    c20: jnp.ndarray
    k: jnp.ndarray
    q: jnp.ndarray
    s: jnp.ndarray

class GPParams(NamedTuple) :
    dev_ls: jnp.ndarray = None
    dev_sig: jnp.ndarray = None
    dev_u_mean: jnp.ndarray = None
    dev_u_var: jnp.ndarray = None
    dev_z: jnp.ndarray = None

    vol_ls: jnp.ndarray = None
    vol_sig: jnp.ndarray = None
    vol_u_mean: jnp.ndarray = None
    vol_u_var: jnp.ndarray = None
    vol_z: jnp.ndarray = None

    c01: jnp.ndarray = None
    c02: jnp.ndarray = None
    c10: jnp.ndarray = None
    c11: jnp.ndarray = None
    c20: jnp.ndarray = None
    k: jnp.ndarray = None
    q: jnp.ndarray = None
    s: jnp.ndarray = None

    sigma_free_x: jnp.ndarray = None
    sigma_free_y: jnp.ndarray = None
    sigma_fix_x: jnp.ndarray = None
    sigma_fix_y: jnp.ndarray = None

class GPWeights(NamedTuple) :
    dev_Kzz: jnp.ndarray
    dev_v: jnp.ndarray
    dev_trace_term: jnp.ndarray
    dev_mahalanobis_term: jnp.ndarray
    dev_M_mat: jnp.ndarray
    dev_Kzz_inv: jnp.ndarray
    dev_logterm: jnp.ndarray

    vol_Kzz: jnp.ndarray
    vol_v: jnp.ndarray
    vol_trace_term: jnp.ndarray
    vol_mahalanobis_term: jnp.ndarray
    vol_M_mat: jnp.ndarray
    vol_Kzz_inv: jnp.ndarray
    vol_logterm: jnp.ndarray

class Params(NamedTuple):
    c01: jnp.ndarray
    c02: jnp.ndarray
    c10: jnp.ndarray
    c11: jnp.ndarray
    c20: jnp.ndarray
    k: jnp.ndarray
    q: jnp.ndarray
    s: jnp.ndarray
    c01_var: jnp.ndarray
    c02_var: jnp.ndarray
    c10_var: jnp.ndarray
    c11_var: jnp.ndarray
    c20_var: jnp.ndarray
    k_var: jnp.ndarray
    q_var: jnp.ndarray
    s_var: jnp.ndarray  
class RawParams(NamedTuple) :
    raw_c01: jnp.ndarray
    raw_c02: jnp.ndarray
    raw_c10: jnp.ndarray
    raw_c11: jnp.ndarray
    raw_c20: jnp.ndarray
    raw_k: jnp.ndarray
    raw_q: jnp.ndarray
    raw_s: jnp.ndarray
    raw_c01_var: jnp.ndarray
    raw_c02_var: jnp.ndarray
    raw_c10_var: jnp.ndarray
    raw_c11_var: jnp.ndarray
    raw_c20_var: jnp.ndarray
    raw_k_var: jnp.ndarray
    raw_q_var: jnp.ndarray
    raw_s_var: jnp.ndarray  


class SyntheticData(NamedTuple) :
    load_array: jnp.ndarray
    u_array: jnp.ndarray
    coords: jnp.ndarray
    cells: jnp.ndarray
    node_type: jnp.ndarray
    dev_inv_array: jnp.ndarray
    vol_inv_array: jnp.ndarray
    invariants_array: jnp.ndarray
    disp_noise_level: float
    load_noise_level: float

class TrainingSetup(NamedTuple) :
    dataset_dir: str
    material_model: str
    save_path: str
    training_mode: str
    n_dev_ip: int
    n_vol_ip: int
    true_material_parameters: jnp.ndarray


class PrecomputedVFMData(NamedTuple) :
    f_neu: jnp.ndarray
    node_type: jnp.ndarray
    F: jnp.ndarray
    dNdX: jnp.ndarray
    dA: jnp.ndarray