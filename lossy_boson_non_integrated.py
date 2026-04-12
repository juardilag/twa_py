import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps'])
def generate_markovian_noise(key, num_steps, dt, kappa):
    """
    Generates pure white noise increments for the explicit cavity integration.
    """
    k_re, k_im = jax.random.split(key)
    # Exact variance for the OU decay over step dt
    exact_variance = 0.25 * (1.0 - jnp.exp(-kappa * dt))
    
    dw_re = jax.random.normal(k_re, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    dw_im = jax.random.normal(k_im, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    
    return dw_re + 1j * dw_im

@jax.jit
def heun_step_coupled_continuous(carry, step_idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins=1):
    """
    Standard Continuous TWA explicit step using Exponential Time Differencing.
    """
    S_curr, alpha_curr = carry
    
    noise_inc = noise_traj[step_idx - 1]
    
    # 1. Exact Linear Propagators (Exponential Time Differencing)
    z = 1j * omega_0 + 0.5 * kappa
    exact_decay = jnp.exp(-z * dt)
    
    # The EXACT analytical integral of the spin drive over the rotating timestep dt
    phi_drive = (1.0 - exact_decay) / z
    
    coupling_strength = g / jnp.sqrt(n_spins)

    # --- 1. PREDICTOR ---
    B_eff_p_x = 0.5 * B_field[0] + 2.0 * coupling_strength * jnp.real(alpha_curr)
    B_eff_p = jnp.array([B_eff_p_x, 0.5 * B_field[1], 0.5 * B_field[2]])
    
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
              
    # Cavity predictor: We use phi_drive instead of dt!
    alpha_pred = alpha_curr * exact_decay - 1j * coupling_strength * S_curr[0] * phi_drive + noise_inc
    
    # --- 2. CORRECTOR ---
    B_eff_c_x = 0.5 * B_field[0] + 2.0 * coupling_strength * jnp.real(alpha_pred)
    B_eff_c = jnp.array([B_eff_c_x, 0.5 * B_field[1], 0.5 * B_field[2]])
    
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
              
    # Cavity corrector: We use phi_drive instead of dt!
    alpha_next = alpha_curr * exact_decay - 1j * coupling_strength * 0.5 * (S_curr[0] + S_pred[0]) * phi_drive + noise_inc

    new_carry = (S_next, alpha_next)
    
    return new_carry, (S_next, alpha_next)

def run_coupled_twa_bundle(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1):
    """
    Computes ONLY the 1D expectation values. Highly optimized for speed.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        k_init_re, k_init_im = jax.random.split(k_samp_alpha)
        vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
        vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
        alpha0 = jnp.sqrt(n_photons_initial) + (vacuum_fluc_re + 1j * vacuum_fluc_im)
        
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        carry_init = (s0, alpha0)
        
        def scan_body(carry, idx):
            return heun_step_coupled_continuous(
                carry, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
            )

        _, (S_traj, alpha_traj) = jax.lax.scan(scan_body, carry_init, jnp.arange(1, num_steps))
        
        return jnp.vstack([s0, S_traj]), jnp.append(alpha0, alpha_traj)

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_S, batch_alpha = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_S, axis=0), jnp.sum(batch_alpha, axis=0)

    total_sum_S = jnp.zeros((num_steps, 3))
    total_sum_alpha = jnp.zeros(num_steps, dtype=jnp.complex64)
    
    n_batches = int(jnp.ceil(n_total / batch_size))

    print(f"Starting Non-Integrated (Markovian) DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="DTWA Expectations"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_sum_S, batch_sum_alpha = process_batch_sum(current_keys)
        total_sum_S += batch_sum_S
        total_sum_alpha += batch_sum_alpha
        
    return total_sum_S / n_total, total_sum_alpha / n_total


def run_coupled_twa_correlations(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1):
    """
    Computes ONLY the 2D two-time Wigner correlation matrices.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        k_init_re, k_init_im = jax.random.split(k_samp_alpha)
        vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
        vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
        alpha0 = jnp.sqrt(n_photons_initial) + (vacuum_fluc_re + 1j * vacuum_fluc_im)
        
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        carry_init = (s0, alpha0)
        
        def scan_body(carry, idx):
            return heun_step_coupled_continuous(
                carry, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
            )

        _, (S_traj, alpha_traj) = jax.lax.scan(scan_body, carry_init, jnp.arange(1, num_steps))
        
        return jnp.vstack([s0, S_traj]), jnp.append(alpha0, alpha_traj)

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_S, batch_alpha = jax.vmap(solve_single_trajectory)(batch_keys)
        
        # 2D Wigner Correlation Matrices ONLY
        # jnp.outer(A, B) creates the matrix where element i,j is A[i]*B[j]
        sum_corr_alpha = jnp.sum(jax.vmap(lambda a: jnp.outer(jnp.conj(a), a))(batch_alpha), axis=0)
        sum_corr_Sx = jnp.sum(jax.vmap(lambda s: jnp.outer(s[:, 0], s[:, 0]))(batch_S), axis=0)
        sum_corr_Sy = jnp.sum(jax.vmap(lambda s: jnp.outer(s[:, 1], s[:, 1]))(batch_S), axis=0)
        sum_corr_Sz = jnp.sum(jax.vmap(lambda s: jnp.outer(s[:, 2], s[:, 2]))(batch_S), axis=0)
        
        return sum_corr_alpha, sum_corr_Sx, sum_corr_Sy, sum_corr_Sz

    # Initialize accumulators for matrices only
    total_corr_alpha = jnp.zeros((num_steps, num_steps), dtype=jnp.complex64)
    total_corr_Sx = jnp.zeros((num_steps, num_steps))
    total_corr_Sy = jnp.zeros((num_steps, num_steps))
    total_corr_Sz = jnp.zeros((num_steps, num_steps))
    
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    for i in tqdm(range(n_batches), desc="DTWA 2D Correlations"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        s_ca, s_cx, s_cy, s_cz = process_batch_sum(current_keys)
        
        total_corr_alpha += s_ca
        total_corr_Sx += s_cx
        total_corr_Sy += s_cy
        total_corr_Sz += s_cz
        
    return {
        "corr_alpha": total_corr_alpha / n_total,
        "corr_Sx": total_corr_Sx / n_total,
        "corr_Sy": total_corr_Sy / n_total,
        "corr_Sz": total_corr_Sz / n_total
    }

def run_time_integrated_correlation(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, tau_steps=2000, batch_size=1000, n_spins=1):
    """
    Computes the 1D time-averaged correlation function C(tau) DIRECTLY in the time domain.
    No FFTs are used. It computes the exact sliding-window average.
    
    tau_steps: The maximum delay time index you want to calculate (e.g., 2000 steps).
               Must be smaller than num_steps.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # The number of points we can actually average over for a given tau
    valid_length = num_steps - tau_steps
    tau_indices = jnp.arange(tau_steps)
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        k_init_re, k_init_im = jax.random.split(k_samp_alpha)
        vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
        vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
        alpha0 = jnp.sqrt(n_photons_initial) + (vacuum_fluc_re + 1j * vacuum_fluc_im)
        
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        carry_init = (s0, alpha0)
        
        def scan_body(carry, idx):
            return heun_step_coupled_continuous(
                carry, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
            )

        _, (_, alpha_traj) = jax.lax.scan(scan_body, carry_init, jnp.arange(1, num_steps))
        alpha_full = jnp.append(alpha0, alpha_traj)
        
        # --- THE EXACT TIME-DOMAIN SLIDING WINDOW (NO FFT) ---
        def compute_C_tau(tau):
            # Grab the unshifted base array of size `valid_length`
            a_base = jax.lax.dynamic_slice_in_dim(alpha_full, 0, valid_length)
            
            # Grab the shifted array of the exact same size
            a_shifted = jax.lax.dynamic_slice_in_dim(alpha_full, tau, valid_length)
            
            # Return the exact time-average for this specific tau
            return jnp.mean(jnp.conj(a_shifted) * a_base)
        
        # Vectorize the exact calculation over all requested delay times
        C_tau = jax.vmap(compute_C_tau)(tau_indices)
        return C_tau

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_C = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_C, axis=0)

    total_C_tau = jnp.zeros(tau_steps, dtype=jnp.complex64)
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Computing Exact Time-Domain Correlation up to tau={tau_steps} steps...")
    for i in tqdm(range(n_batches), desc="Autocorrelation Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        total_C_tau += process_batch_sum(current_keys)
        
    C_wigner_tau = total_C_tau / n_total
    
    # --- EXACT LINEAR RESPONSE (VACUUM SUBTRACTION) ---
    tau_time = tau_indices * dt
    chi_tau = jnp.exp((1j * omega_0 - 0.5 * kappa) * tau_time)
    
    # Physical Correlation: <a+(tau) a(0)> = C_Wigner - 0.5 * <[a(tau), a+(0)]>
    C_physical_tau = C_wigner_tau - 0.5 * chi_tau
    
    return tau_time, C_physical_tau, C_wigner_tau

@jax.jit
def compute_spectrum(C_tau, tau_grid, omega_grid):
    """
    Computes the emission spectrum using a direct Riemann sum over an arbitrary frequency grid.
    Bypasses FFT to allow for infinite sub-grid frequency resolution.
    """
    dt = tau_grid[1] - tau_grid[0]
    
    # Optional but highly recommended: Apply an apodization window.
    # If C_tau hasn't completely decayed to 0 at the end of the array, 
    # the sharp cut-off will cause artificial "ringing" in your spectrum.
    window = jnp.hanning(C_tau.shape[0])
    C_tau_windowed = C_tau * window
    
    def compute_single_omega(omega):
        # 1. Evaluate the integrand: C(tau) * exp(i * w * tau)
        integrand = C_tau_windowed * jnp.exp(1j * omega * tau_grid)
        
        # 2. Riemann Sum (dt * sum)
        one_sided_integral = jnp.sum(integrand) * dt
        
        # 3. Multiply by 2 and take the Real part to reconstruct the full symmetric integral
        return 2.0 * jnp.real(one_sided_integral)

    # Vectorize the integration perfectly across the entire custom frequency array
    spectrum = jax.vmap(compute_single_omega)(omega_grid)
    
    return spectrum