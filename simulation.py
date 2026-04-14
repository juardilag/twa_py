import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"   
import jax
import jax.numpy as jnp
from lossy_boson import solve_dynamics_vacuum, get_initial_state
from lossy_boson_integrated import run_integrated_twa_bundle
from lossy_boson_non_integrated import run_coupled_twa_bundle
import time


def run_normalized_simulation(g_ratio, kappa_ratio, B_field_unit, v_init, tau_max, omega_0, num_steps, n_photons=1, n_spins=1, boson_truncation=30, run_qutip=True):
    """
    Runs QuTiP, Integrated DTWA, and Explicit Coupled DTWA simulations 
    using normalized parameters for N spins, and calculates the emission spectrum.
    
    Returns:
        dict: Contains t_grid, normalized QuTiP expectation values [-1, 1], 
              normalized TWA results [-1, 1], and the spin emission spectrum.
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
    rho0 = get_initial_state(v_init, n_photons, N_spins=n_spins, N_boson=boson_truncation)
    
    # 4. Run QuTiP Solver
    print("Running Exact Diagonalization")
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
        N_boson=boson_truncation
    )
    qutip_x_raw = res.expect[0]
    qutip_y_raw = res.expect[1]
    qutip_z_raw = res.expect[2]
    qutip_boson_raw = res.expect[3]
    
    # 5. Setup TWA PRNG Keys
    n_trajectories = 25_000 
    master_key = jax.random.PRNGKey(42)
    key_integrated, key_coupled = jax.random.split(master_key)
    
    keys_int = jax.random.split(key_integrated, n_trajectories)
    keys_coup = jax.random.split(key_coupled, n_trajectories)

    # 6. Run Integrated (Non-Markovian) TWA Simulation
    print("Running Integrated (Non-Markovian) DTWA")
    twa_integrated_raw = run_integrated_twa_bundle(
        keys=keys_int, 
        t_grid=t_grid, 
        omega_0=omega_0, 
        kappa=kappa, 
        B_field=B_scaled, 
        g=g, 
        n_photons_initial=n_photons,
        initial_direction=jnp.array(v_init),
        batch_size=25_000,
        n_spins=n_spins
    )

    # 7. Run Explicit Coupled TWA Simulation
    print("Running Non-Integrated (Markovian) DTWA")
    twa_coupled_S, twa_coupled_alpha = run_coupled_twa_bundle(
        keys=keys_coup, 
        t_grid=t_grid, 
        omega_0=omega_0, 
        kappa=kappa, 
        B_field=B_scaled, 
        g=g, 
        n_photons_initial=n_photons,
        initial_direction=jnp.array(v_init),
        batch_size=25_000,
        n_spins=n_spins
    )

    # 9. Organize and Normalize Data to intensive magnetization [-1, 1]
    qutip_norm = n_spins / 2.0
    twa_norm = n_spins
    coupled_boson_num = jnp.abs(twa_coupled_alpha)**2 - 0.5

    return {
        "t_grid": t_grid,
        "omega_0": omega_0,
        "n_spins": n_spins,
        "qutip": {
            "expect_x": qutip_x_raw / qutip_norm,
            "expect_y": qutip_y_raw / qutip_norm,
            "expect_z": qutip_z_raw / qutip_norm,
            "boson_num": qutip_boson_raw
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


@jax.jit
def compute_curve_rmse(qutip_x, qutip_y, qutip_z, twa_x, twa_y, twa_z):
    """
    Computes the Root Mean Squared Error (RMSE) over the time grid 
    between the QuTiP exact solution and a TWA solution.
    Fully JAX-compliant and JIT-compiled.
    """
    diff_x = qutip_x - twa_x
    diff_y = qutip_y - twa_y
    diff_z = qutip_z - twa_z
    
    # Average the squared differences across time and components, then root
    mse = jnp.mean(diff_x**2 + diff_y**2 + diff_z**2)
    
    # Return the pure JAX array instead of casting to Python float()
    return jnp.sqrt(mse)


def run_1d_scaling_benchmark(n_spins_list, boson_truncation, g_ratio, kappa_ratio, 
                             B_field_unit, v_init, tau_max, omega_0, num_steps, n_photons=1):
    """
    Benchmarks computation time and accuracy vs. QuTiP as a function of n_spins.
    Uses pure JAX arrays and .at[].set() for in-place updates.
    """
    # 1. Scale parameters relative to omega_0
    kappa = kappa_ratio * omega_0
    g = g_ratio * omega_0
    B_scaled = jnp.array(B_field_unit) * omega_0 
    
    t_max = tau_max / omega_0
    t_grid = jnp.linspace(0, t_max, num_steps)
    
    # 2. Setup TWA PRNG Keys
    n_trajectories = 50_000 
    master_key = jax.random.PRNGKey(42)
    key_integrated, key_coupled = jax.random.split(master_key)
    keys_int = jax.random.split(key_integrated, n_trajectories)
    keys_coup = jax.random.split(key_coupled, n_trajectories)

    # 3. Initialize storage arrays as JAX arrays
    num_points = len(n_spins_list)
    times_qutip = jnp.zeros(num_points)
    times_integrated = jnp.zeros(num_points)
    times_coupled = jnp.zeros(num_points)
    
    errors_integrated = jnp.zeros(num_points)
    errors_coupled = jnp.zeros(num_points)

    # 4. Run Benchmark Loop
    for i, n_spins in enumerate(n_spins_list):
        print(f"Benchmarking: n_spins={n_spins} (Fixed bosons={boson_truncation})")
        
        # Normalization factors
        qutip_norm = n_spins / 2.0
        twa_norm = n_spins
        
        # --- QuTiP Timing & Execution ---
        rho0 = get_initial_state(v_init, n_photons, N_spins=n_spins, N_boson=boson_truncation)
        start_time = time.perf_counter()
        res = solve_dynamics_vacuum(
            Bx=B_scaled[0], By=B_scaled[1], Bz=B_scaled[2], 
            wa=omega_0, g=g, kappa=kappa, times=t_grid, 
            rho0=rho0, N_spins=n_spins, N_boson=boson_truncation
        )
        # Use JAX .at[idx].set() syntax
        times_qutip = times_qutip.at[i].set(time.perf_counter() - start_time)
        
        # Extract and normalize QuTiP curves
        qx = jnp.array(res.expect[0]) / qutip_norm
        qy = jnp.array(res.expect[1]) / qutip_norm
        qz = jnp.array(res.expect[2]) / qutip_norm
        
        # --- Integrated DTWA Timing & Execution ---
        start_time = time.perf_counter()
        twa_integrated_raw = run_integrated_twa_bundle(
            keys=keys_int, t_grid=t_grid, omega_0=omega_0, kappa=kappa, 
            B_field=B_scaled, g=g, n_photons_initial=n_photons,
            initial_direction=jnp.array(v_init), batch_size=25_000, n_spins=n_spins
        )
        twa_integrated_raw.block_until_ready() 
        times_integrated = times_integrated.at[i].set(time.perf_counter() - start_time)
        
        # Extract, normalize, and compare Integrated DTWA
        ix, iy, iz = twa_integrated_raw[:, 0]/twa_norm, twa_integrated_raw[:, 1]/twa_norm, twa_integrated_raw[:, 2]/twa_norm
        err_int = compute_curve_rmse(qx, qy, qz, ix, iy, iz)
        errors_integrated = errors_integrated.at[i].set(err_int)
        
        # --- Coupled DTWA Timing & Execution ---
        start_time = time.perf_counter()
        twa_coupled_S, twa_coupled_alpha = run_coupled_twa_bundle(
            keys=keys_coup, t_grid=t_grid, omega_0=omega_0, kappa=kappa, 
            B_field=B_scaled, g=g, n_photons_initial=n_photons,
            initial_direction=jnp.array(v_init), batch_size=25_000, n_spins=n_spins
        )
        twa_coupled_S.block_until_ready()
        times_coupled = times_coupled.at[i].set(time.perf_counter() - start_time)
        
        # Extract, normalize, and compare Coupled DTWA
        cx, cy, cz = twa_coupled_S[:, 0]/twa_norm, twa_coupled_S[:, 1]/twa_norm, twa_coupled_S[:, 2]/twa_norm
        err_coup = compute_curve_rmse(qx, qy, qz, cx, cy, cz)
        errors_coupled = errors_coupled.at[i].set(err_coup)

    return times_qutip, times_integrated, times_coupled, errors_integrated, errors_coupled