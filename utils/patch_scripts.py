import os

def patch_file(input_file, output_file, is_traction=False):
    with open(input_file, 'r') as f:
        content = f.read()

    # 1. Update arguments
    old_args = "parser.add_argument('--model_path', type=str, default=\"20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0\")\n    # parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[2, 4, 6, 8])\n    parser.add_argument('--n_sample', type=int, default=128)"
    if old_args not in content and is_traction:
        old_args = "parser.add_argument('--model_path', type=str, default=\"20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0\")\n    # parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[2, 4, 6, 8])\n    parser.add_argument('--n_sample', type=int, default=128)"
        
    # fallback
    if "parser.add_argument('--n_sample'" in content:
        content = content.replace("parser.add_argument('--n_sample', type=int, default=128)",
                                  "parser.add_argument('--distilled_dir', type=str, required=True)\n    parser.add_argument('--material_model', type=str, required=True)\n    parser.add_argument('--n_sample', type=int, default=128)")

    # 2. Update save path
    content = content.replace("save_path = analysis_dir / case_name",
                              "save_path = analysis_dir / Path(args.distilled_dir).name")

    # 3. Replace GP model loading with flow_samples
    gp_load_str = """    I_obs_all = np.load(extraction_result_dir / case_name / "I_obs_all.npy")
    I_z = np.load(extraction_result_dir / case_name / "I_z.npy")
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    best_raw_params = np.load(extraction_result_dir / case_name / "best_params.npy", allow_pickle=True).item()
    best_raw_params = GPRawParams(**best_raw_params)
    model = SparseHyperelasticityGP(best_raw_params, I_z, min_dev, min_vol, max_dev, max_vol)
    model.params = model.load_params(best_raw_params)
    
    model.gpweight = model.precompute_weights(best_raw_params)"""
    
    flow_load_str = """    flow_samples_path = os.path.join(args.distilled_dir, "flow_samples.npy")
    flow_samples = np.load(flow_samples_path)
    np.random.seed(42)
    sample_indices = np.random.choice(len(flow_samples), n_sample, replace=False)
    selected_samples = flow_samples[sample_indices]"""
    content = content.replace(gp_load_str, flow_load_str)

    # 4. In simulation loop, replace lambda
    if not is_traction:
        old_lambda = "material_model_piola_stress=lambda f: model.piola(fto3x3(f), subkey)[:2, :2]"
    else:
        old_lambda = "material_model_piola_stress=lambda f: model.piola(fto3x3(f), key_piola)[:2, :2]"

    new_lambda = """material_model_piola_stress=lambda f: mat.P(fto3x3(f))[:2, :2]"""
    content = content.replace(old_lambda, new_lambda)

    # 5. Insert material construction before problem_pred definition
    prob_def = "            problem_pred = HyperElasticity("
    mat_def = """            params = selected_samples[i]
            if args.material_model == "isihara":
                c10, c01, c20, d1 = params
                mat = get_material(args.material_model, c10=c10, c01=c01, c20=c20, d1=d1)
            elif args.material_model == "gmr":
                c10, c01, c20, c02, c11, d1 = params
                mat = get_material(args.material_model, c10=c10, c01=c01, c20=c20, c02=c02, c11=c11, d1=d1)
            else:
                mat = get_material(args.material_model)
                
            problem_pred = HyperElasticity("""
    content = content.replace(prob_def, mat_def)

    with open(output_file, 'w') as f:
        f.write(content)


patch_file('forward_fem_piola_sample.py', 'forward_fem_distilled_piola_sample.py', is_traction=False)
patch_file('forward_fem_piola_traction_sample.py', 'forward_fem_distilled_piola_traction_sample.py', is_traction=True)
print("Files patched successfully!")
