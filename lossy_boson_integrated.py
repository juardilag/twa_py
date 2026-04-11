import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm
 
@jax.jit
def generate_complete_noise(key, t_grid, omega_0, kappa, g, n_photons_initial=0.0, n_spins=1):
    dt = t_grid[1] - t_grid[0]
    # Necesitamos 4 llaves: 2 para el estado inicial de la cavidad, 2 para el baño
    k_init_re, k_init_im, k_re, k_im = jax.random.split(key, 4)
    
    # 1. Ruido Transitorio (ESTADO COHERENTE CORRECTO)
    # alpha_0 = Campo Medio + Fluctuaciones del Vacío (Wigner)
    mean_field = jnp.sqrt(n_photons_initial) # Asume fase 0 (campo coherente real)
    vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
    vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
    
    # El estado inicial es una Gaussiana centrada en el campo medio
    alpha_0 = mean_field + (vacuum_fluc_re + 1j * vacuum_fluc_im)
    transient_alpha = alpha_0 * jnp.exp(-(1j * omega_0 + 0.5 * kappa) * t_grid)
    
    # 2. Ruido Estacionario del Baño (EXACT Ornstein-Uhlenbeck Integrator)
    # Pre-calculate the exact analytical decay and variance for the time step
    exact_decay = jnp.exp(-(1j * omega_0 + 0.5 * kappa) * dt)
    exact_variance = 0.25*(1.0 - jnp.exp(-kappa * dt)) 
    
    # Generate the noise increments using the exact variance
    dw_re = jax.random.normal(k_re, shape=t_grid.shape) * jnp.sqrt(exact_variance)
    dw_im = jax.random.normal(k_im, shape=t_grid.shape) * jnp.sqrt(exact_variance)
    exact_noise_increments = dw_re + 1j * dw_im
    
    # The exact, unconditionally stable EOM step
    def exact_cavity_step(alpha_prev, noise_inc):
        alpha_next = alpha_prev * exact_decay + noise_inc
        return alpha_next, alpha_next

    _, bath_alpha = jax.lax.scan(exact_cavity_step, 0j, exact_noise_increments)
    
    # 3. Campo Total de la Cavidad
    alpha_total = transient_alpha + bath_alpha
    phi_noise = 2.0 * jnp.real(alpha_total)
    
    # Xi = 2 * (g/sqrt(N)) * phi (Fuerza sobre el espín con escalamiento termodinámico)
    xi_x = 2.0 * (g / jnp.sqrt(n_spins)) * phi_noise
    return jnp.zeros((t_grid.shape[0], 3)).at[:, 0].set(xi_x)


def compute_memory_kernel(tau_grid, omega_0, kappa, g=1.0, n_spins=1):
    """
    Theoretical Memory Kernel: gamma^R(t) magnitude.
    Captures the 4 * (g^2 / N) * exp(-kappa*t/2) * sin(w0*t) structure.
    """
    gamma_kernel = 4.0 * jnp.exp(-0.5 * kappa * tau_grid) * jnp.sin(omega_0 * tau_grid)
    # El acoplamiento al cuadrado se escala como 1/N
    return ((g**2) / n_spins) * gamma_kernel

@jax.jit
def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, 
                            noise_traj, B_field, dt, n_spins=1):
    """
    Campo efectivo calibrado para DTWA.
    Factores: Ext=1.0, Noise=0.5, Memory=0.5
    (n_spins se incluye en la firma para mantener simetría, el escalamiento 
    ya está embebido en noise_traj y gamma_kernel).
    """
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices
    gamma_causal = jnp.where(lag_indices > 0, 
                             jnp.take(gamma_kernel, lag_indices, mode='fill', fill_value=0.0), 
                             0.0)

    # Memoria a 0.5: Compensa el 2.0 del integrador. 
    # history_array crece con N, gamma_causal decae con 1/N -> La fuerza resultante es intensiva O(1)
    memory_x = 0.5*(jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    
    # Ruido a 0.5: Recupera la escala física. (Ya fue escalado por 1/sqrt(N))
    xi_t = noise_traj[jnp.clip(step_idx, 0, noise_traj.shape[0]-1)]
    xi_field = 0.5*xi_t[0]
    
    # Campo Externo a 1.0 para mantener las frecuencias en resonancia
    eff_field_x = 0.5*B_field[0] + xi_field - memory_x
    
    return jnp.array([eff_field_x, 0.5*B_field[1], 0.5*B_field[2]])

@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt, n_spins=1):
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]

    # --- PREDICTOR ---
    B_eff_p = compute_effective_field(S_curr, state_trajectory, curr_idx, gamma_kernel, noise_traj, B_field, dt, n_spins)
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    # The factor of 2.0 remains exact because we are tracking Sigma (length N), not S (length N/2)
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
    
    # --- CORRECTOR ---
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    B_eff_c = compute_effective_field(S_pred, traj_with_pred, step_idx, gamma_kernel, noise_traj, B_field, dt, n_spins)
    
    # Average the effective magnetic fields directly!
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
              
    return state_trajectory.at[step_idx].set(S_next), S_next


def run_integrated_twa_bundle(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, coupling_type, batch_size=1000, n_spins=1):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # Kernel requires n_spins to correctly scale the retardation 1/N
    gamma_kernel_fine = compute_memory_kernel(t_grid, omega_0, kappa, g, n_spins)
    
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        # New collective sampling!
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction, n_spins)
        
        # Noise explicitly incorporates the 1/sqrt(N) scaling
        noise_traj = generate_complete_noise(k_noise, t_grid, omega_0, kappa, g, n_photons_initial, n_spins)
        
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        def scan_body(carry, idx):
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, B_field, dt, n_spins)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_trajs, axis=0)

    total_sum = jnp.zeros((num_steps, 3))
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Starting DTWA for N={n_spins} spins: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="DTWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        total_sum = total_sum + process_batch_sum(current_keys)
        
    # The output array naturally scales from -N to N. 
    # If you want it normalized to [-1, 1] for plotting against QuTiP, divide by (n_total * n_spins) outside the function.
    return total_sum / n_total