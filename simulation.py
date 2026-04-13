import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"   
import jax
import jax.numpy as jnp
from lossy_boson import solve_dynamics_vacuum, get_initial_state
from lossy_boson_integrated import run_integrated_twa_bundle
from lossy_boson_non_integrated import run_coupled_twa_bundle, run_time_integrated_light_correlation, compute_spectrum


def run_normalized_simulation(g_ratio, kappa_ratio, B_field_unit, v_init, tau_max, omega_0, n_photons_initial, num_steps, n_spins=1, N_boson=30, run_qutip=True):
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
    rho0 = get_initial_state(v_init, n_photons_initial, N_spins=n_spins, N_boson=N_boson)
    
    # 4. Run QuTiP Solver (Optional)
    if run_qutip:
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
        qutip_x_raw = res.expect[0]
        qutip_y_raw = res.expect[1]
        qutip_z_raw = res.expect[2]
        qutip_boson_raw = res.expect[3]
    else:
        print("--- Skipping Exact QuTiP Simulation ---")
        # Use NaN arrays so downstream Matplotlib plots won't crash
        null_array = jnp.full_like(t_grid, jnp.nan)
        qutip_x_raw = null_array
        qutip_y_raw = null_array
        qutip_z_raw = null_array
        qutip_boson_raw = null_array
    
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
        batch_size=25_000,
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
        batch_size=25_000,
        n_spins=n_spins
    )

    t_grid_corr = jnp.linspace(0, t_max, num_steps)
    dt = t_grid_corr[1] - t_grid_corr[0]
    
    # 8. Run Spin Correlation and Spectrum
    print("--- Computing Emission Spectrum (Spin Dipole) ---")
    
    # Allow tau to run up to 90% of the total simulation, but cap at 2500
    safe_tau_steps = min(2500, int(num_steps * 0.9))
    
    tau_time, C_physical_tau = run_time_integrated_light_correlation(
        keys=keys_coup,                 
        t_grid=t_grid_corr, 
        omega_0=omega_0, 
        kappa=kappa, 
        B_field=B_scaled, 
        g=g, 
        n_photons_initial=n_photons_initial, 
        initial_direction=v_init,
        tau_steps=safe_tau_steps,
        batch_size=25_000,               
        n_spins=n_spins
    )

    tau_steps = safe_tau_steps
    tau_indices = jnp.arange(tau_steps)

    # --- THE ROBUST ENVELOPE & FALLBACK THRESHOLD ---
    noise_threshold = 0.05
    
    # 1. Create a smooth envelope to ignore oscillations and random Monte Carlo spikes
    smoothing_window = 30
    kernel = jnp.ones(smoothing_window) / smoothing_window
    envelope = jnp.convolve(jnp.abs(C_physical_tau), kernel, mode='same')

    # 2. Apply threshold to the smooth envelope
    active_mask = envelope > noise_threshold
    last_active_idx = jnp.max(tau_indices * active_mask)

    # 3. Define buffers
    buffer_steps = 20
    taper_steps = 30 # Generous taper to ensure a smooth transition to 0.0

    # 4. THE SAFETY NET: 
    # Force the cutoff to happen early enough so the taper fully finishes before the array ends.
    # This prevents the cliff even if the signal is still huge at the end of the simulation.
    max_allowable_cutoff = tau_steps - taper_steps - 1
    cutoff_idx = jnp.minimum(last_active_idx + buffer_steps, max_allowable_cutoff)

    # 5. Build dynamic window
    def build_dynamic_window(idx):
        dist = idx - cutoff_idx
        return jnp.where(
            dist <= 0,
            1.0, # Keep physics untouched
            jnp.where(
                dist < taper_steps,
                jnp.cos(jnp.pi / 2.0 * (dist / taper_steps))**2, # Soften the cut
                0.0 # Total silence for padding
            )
        )

    dynamic_window = jax.vmap(build_dynamic_window)(tau_indices)

    C_clean = C_physical_tau * dynamic_window
    
    # 6. Zero-Pad and Transform
    pad_length = 5000
    C_padded = jnp.pad(C_clean, (0, pad_length))
    tau_padded = jnp.arange(tau_steps + pad_length) * dt

    omega_zoom = jnp.linspace(omega_0 - 1, omega_0 + 1, num_steps)
    spectrum_zoom = compute_spectrum(C_padded, tau_padded, omega_zoom)
    
    # 7. Clean Baseline Noise Floor
    real_spectrum = jnp.real(spectrum_zoom)

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
        },
        "spectrum": {
            "tau_time": tau_padded,
            "C_spin_tau": C_padded,
            "omega": omega_zoom,
            "S_omega": real_spectrum
        }
    }
