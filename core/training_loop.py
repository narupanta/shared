def stochastic_training_loop(max_iter, save_path, log_file_path) :
    pass

def deterministic_training_loop() :
    pass
    # warmup_sq = optax.linear_schedule(
    # init_value=1e-3, 
    # end_value=5e-3, 
    # transition_steps=2000
    # )

    # decay_sq = optax.cosine_decay_schedule(
    #     init_value=5e-3, 
    #     decay_steps=18000, # Total steps minus warmup
    #     alpha=0.1          # End at 10% of the peak value
    # )

    # # 2. Join them
    # schedule = optax.join_schedules(
    #     schedules=[warmup_sq, decay_sq],
    #     boundaries=[2000]
    # )

    # # 3. Use it in the optimizer
    # opt = optax.adam(learning_rate=schedule)
    # opt_state = opt.init(params)
    # max_iter = 30000
    # def det_loss(u_array, loads, p, coords, cells, node_type) :
    #     model.params = model.load_params(p)
    #     model.gpweight = model.precompute_weights(p)
    #     piola_func = lambda f: model.piola_det(f)
    #     free_loss, fix_loss = total_physical_loss(u_array, loads, piola_func, coords, cells, node_type)
    #     return jnp.sum(free_loss**2) + jnp.sum(fix_loss**2), (jnp.sum(free_loss**2), jnp.sum(fix_loss**2))


    # loss_and_grad_deterministic = jax.jit(jax.value_and_grad(
    #     lambda p: det_loss(u_array, loads, p, coords, cells, node_type),
    #     has_aux=True)
    # )

    # det_losses = []
    # deterministic_pbar = tqdm(range(max_iter), desc="Deterministic Training", unit="step")
    # best_det_loss = float('inf')
    # for step in deterministic_pbar :
    #     (loss, (free_loss, fix_loss)), grads = loss_and_grad_deterministic(params)

    #     updates, opt_state = opt.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)
    #     det_losses.append(loss)
    #     # save best material parameters
    #     if loss < best_det_loss:
    #         best_det_loss = loss
    #         best_params = params
    #         with open(os.path.join(save_path, "best_mat_params.npy"), "wb") as f:
    #             jnp.save(f, best_params._asdict())


    #     if step % 50 == 0:
    #         # Update the progress bar postfix with current metrics
    #         # This shows up right next to the time left
    #         mat_params = model.load_params(best_params)
    #         deterministic_pbar.set_postfix({
    #             "loss": f"{loss:.4f}",
    #             "free_loss": f"{free_loss:.4f}",
    #             "fix_loss": f"{fix_loss:.4f}",
    #             "c01": f"{mat_params.c01:.4f}",
    #             "c02": f"{mat_params.c02:.4f}",
    #             "c10": f"{mat_params.c10:.4f}",
    #             "c11": f"{mat_params.c11:.4f}",
    #             "c20": f"{mat_params.c20:.4f}",
    #             "k": f"{mat_params.k:.4f}",
    #             "q": f"{mat_params.q:.4f}",
    #             "s": f"{mat_params.s:.4f}",
    #         })

    #         # --- Your existing logging logic ---
    #         # Note: Using pbar.write() instead of print() prevents 
    #         # the progress bar from breaking into multiple lines.
    #         log_message = (
    #             f"step {step:04d} | loss={loss:.6f} | "
    #             f"free_loss={free_loss:.6f} | fix_loss={fix_loss:.6f}\n"
    #         )