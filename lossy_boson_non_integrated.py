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