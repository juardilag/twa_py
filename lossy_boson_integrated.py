import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps'])
def generate_complete_noise(key, num_steps, dt, omega_0, kappa, g, n_photons_initial=0.0, n_spins=1):
    """
    Generates the exact analytical Ornstein-Uhlenbeck noise trajectory.
    Optimized with static shape sizing for rapid JIT compilation.
    """
    k_init_re, k_init_im, k_re, k_im = jax.random.split(key, 4)
    
    # 1. Transient Noise (Correct Coherent State + Wigner Vacuum)
    mean_field = jnp.sqrt(n_photons_initial) 
    vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
    vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
    alpha_0 = mean_field + (vacuum_fluc_re + 1j * vacuum_fluc_im)
    
    t_grid = jnp.arange(num_steps) * dt
    transient_alpha = alpha_0 * jnp.exp(-(1j * omega_0 + 0.5 * kappa) * t_grid)
    
    # 2. Stationary Bath Noise (Exact Integrator)
    exact_decay = jnp.exp(-(1j * omega_0 + 0.5 * kappa) * dt)
    exact_variance = 0.25 * (1.0 - jnp.exp(-kappa * dt)) 
    
    dw_re = jax.random.normal(k_re, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    dw_im = jax.random.normal(k_im, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    exact_noise_increments = dw_re + 1j * dw_im
    
    def exact_cavity_step(alpha_prev, noise_inc):
        alpha_next = alpha_prev * exact_decay + noise_inc
        return alpha_next, alpha_next

    _, bath_alpha = jax.lax.scan(exact_cavity_step, 0j, exact_noise_increments)
    
    # 3. Total Cavity Field and Thermodynamic Scaling
    alpha_total = transient_alpha + bath_alpha
    phi_noise = 2.0 * jnp.real(alpha_total)
    
    xi_x = 2.0 * (g / jnp.sqrt(n_spins)) * phi_noise
    
    # Return directly as a (num_steps, 3) shaped array to match spin vector dimensions
    return jnp.zeros((num_steps, 3)).at[:, 0].set(xi_x)


@jax.jit
def compute_memory_kernel(t_grid, omega_0, kappa, g, n_spins):
    """
    Theoretical Memory Kernel magnitude.
    Pre-computed outside the scan loop since it is entirely deterministic.
    """
    gamma_kernel = 4.0 * jnp.exp(-0.5 * kappa * t_grid) * jnp.sin(omega_0 * t_grid)
    return ((g**2) / n_spins) * gamma_kernel


@jax.jit
def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, noise_traj, B_field, dt):
    """
    Computes the effective non-Markovian field.
    Optimized the causal mask to avoid dynamic JAX shape errors.
    """
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices
    
    # XLA-Optimized Causal Masking: zero out future states without dynamic slicing
    valid_mask = lag_indices > 0
    safe_lag = jnp.where(valid_mask, lag_indices, 0)
    gamma_causal = jnp.where(valid_mask, gamma_kernel[safe_lag], 0.0)

    # Memory integration (Convolution)
    memory_x = 0.5 * (jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    
    # The noise is already pre-scaled thermodynamically
    xi_field = 0.5 * noise_traj[step_idx, 0]
    
    eff_field_x = 0.5 * B_field[0] + xi_field - memory_x
    
    return jnp.array([eff_field_x, 0.5 * B_field[1], 0.5 * B_field[2]])


@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    """
    Predictor-Corrector step for the integro-differential equation.
    """
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]

    # --- PREDICTOR ---
    B_eff_p = compute_effective_field(S_curr, state_trajectory, curr_idx, gamma_kernel, noise_traj, B_field, dt)
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
    
    # --- CORRECTOR ---
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    B_eff_c = compute_effective_field(S_pred, traj_with_pred, step_idx, gamma_kernel, noise_traj, B_field, dt)
    
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
              
    return state_trajectory.at[step_idx].set(S_next), S_next


def run_integrated_twa_bundle(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1):
    """
    The main execution bundle. Generates noise locally and maps the trajectories.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    gamma_kernel_fine = compute_memory_kernel(t_grid, omega_0, kappa, g, n_spins)
    
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction, n_spins)
        noise_traj = generate_complete_noise(k_noise, num_steps, dt, omega_0, kappa, g, n_photons_initial, n_spins)
        
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        def scan_body(carry, idx):
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, B_field, dt)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_trajs, axis=0)

    total_sum = jnp.zeros((num_steps, 3))
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Starting Integrated (Non-Markovian) DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="Integrated DTWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        total_sum += process_batch_sum(current_keys)
        
    return total_sum / n_total