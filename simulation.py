import jax
import jax.numpy as jnp
from lossy_boson import solve_dynamics_vacuum, get_initial_state
from lossy_boson_integrated import run_integrated_twa_bundle
from lossy_boson_non_integrated import run_coupled_twa_bundle

def run_normalized_simulation(g_ratio, kappa_ratio, B_field_unit, v_init, tau_max, omega_0, n_photons_initial, num_steps, n_spins=1, N_boson=30, coupling='full'):
    """
    Runs QuTiP, Integrated DTWA, and Explicit Coupled DTWA simulations 
    using normalized parameters for N spins.
    
    Returns:
        dict: Contains t_grid, normalized QuTiP expectation values [-1, 1], 
              and normalized TWA results [-1, 1].
    """
    # 1. Scale parameters relative to omega_0
    kappa = kappa_ratio * omega_0
    g = g_ratio * omega_0
    
    # B_field input is treated as the relative magnitude/direction
    B_scaled = jnp.array(B_field_unit) * omega_0 
    
    # 2. Setup Time Grid (Dimensionless Scaling)
    t_max = tau_max / omega_0
    t_grid = jnp.linspace(0, t_max, num_steps)
    
    # 3. Initial State (Requires both spin count and boson truncation)
    rho0 = get_initial_state(v_init, n_photons_initial, N_spins=n_spins, N_boson=N_boson)
    
    # 4. Run QuTiP Solver
    print("--- Running Exact QuTiP Simulation ---")
    res = solve_dynamics_vacuum(
        Bx=B_scaled[0], 
        By=B_scaled[1], 
        Bz=B_scaled[2], 
        wa=omega_0, 
        g=g, 
        kappa=kappa, 
        times=t_grid, 
        rho0=rho0, 
        N_spins=n_spins,
        N_boson=N_boson
    )
    
    # 5. Setup TWA PRNG Keys
    n_trajectories = 50_000 
    master_key = jax.random.PRNGKey(42)
    key_integrated, key_coupled = jax.random.split(master_key)
    
    keys_int = jax.random.split(key_integrated, n_trajectories)
    keys_coup = jax.random.split(key_coupled, n_trajectories)

    # 6. Run Integrated (Non-Markovian) TWA Simulation
    print("--- Running Integrated (Non-Markovian) DTWA ---")
    twa_integrated_raw = run_integrated_twa_bundle(
        keys=keys_int, 
        t_grid=t_grid, 
        omega_0=omega_0, 
        kappa=kappa, 
        B_field=B_scaled, 
        g=g, 
        n_photons_initial=n_photons_initial,
        initial_direction=jnp.array(v_init),
        coupling_type=coupling,
        batch_size=10_000,
        n_spins=n_spins
    )

    # 7. Run Explicit Coupled TWA Simulation
    print("--- Running Explicit Coupled DTWA ---")
    twa_coupled_S, twa_coupled_alpha = run_coupled_twa_bundle(
        keys=keys_coup, 
        t_grid=t_grid, 
        omega_0=omega_0, 
        kappa=kappa, 
        B_field=B_scaled, 
        g=g, 
        n_photons_initial=n_photons_initial,
        initial_direction=jnp.array(v_init),
        batch_size=10_000,
        n_spins=n_spins
    )

    # 8. Organize and Normalize Data to intensive magnetization [-1, 1]
    # QuTiP uses jmat where eigenvalues range from -S to S (where S = n_spins / 2.0)
    qutip_norm = n_spins / 2.0
    
    # DTWA generates vectors of maximum length N (sum of N spins of length 1)
    twa_norm = n_spins
    
    # Extract Wigner photon number mapping: <n> = <|alpha|^2> - 1/2
    coupled_boson_num = jnp.abs(twa_coupled_alpha)**2 - 0.5

    return {
        "t_grid": t_grid,
        "omega_0": omega_0,
        "n_spins": n_spins,
        "qutip": {
            "expect_x": res.expect[0] / qutip_norm,
            "expect_y": res.expect[1] / qutip_norm,
            "expect_z": res.expect[2] / qutip_norm,
            "boson_num": res.expect[3]
        },
        "twa_integrated": {
            "expect_x": twa_integrated_raw[:, 0] / twa_norm,
            "expect_y": twa_integrated_raw[:, 1] / twa_norm,
            "expect_z": twa_integrated_raw[:, 2] / twa_norm
        },
        "twa_coupled": {
            "expect_x": twa_coupled_S[:, 0] / twa_norm,
            "expect_y": twa_coupled_S[:, 1] / twa_norm,
            "expect_z": twa_coupled_S[:, 2] / twa_norm,
            "boson_num": coupled_boson_num
        }
    }