import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm.auto import tqdm

def setup_noise_parameters(t_grid, omega_0, kappa, kBT):
    """
    Sets up the noise transfer matrix to generate the pure normalized field phi_noise.
    Removed the 'g' parameter to prevent double-scaling.
    """
    num_omega = 10_000 
    omega_grid = jnp.linspace(omega_0 - 20*kappa, omega_0 + 20*kappa, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    lorentzian = (kappa / (2.0 * jnp.pi)) / ((omega_grid - omega_0)**2 + (kappa/2.0)**2)
    
    safe_omega = jnp.maximum(omega_grid, 1e-12)
    x = safe_omega / (2.0 * kBT)
    thermal_factor = 1.0 / jnp.tanh(x)
    
    # Generate the pure field (a + a_dag) with variance 1 at T=0
    amplitude = jnp.sqrt(lorentzian * thermal_factor * d_omega) 
    
    time_evolution = jnp.exp(-1j * jnp.outer(t_grid, omega_grid))
    transfer_matrix = time_evolution * amplitude[None, :]
    
    return transfer_matrix

@jax.jit
def generate_complete_noise(key, t_grid, omega_0, kappa, g, kBT):
    k1, k2 = jax.random.split(key)
    
    # Phenomenological ZPE Scaling
    # lambda_zpe = 1.0 is the exact mathematical derivation (Causes ZPE Leakage)
    
    radius = jnp.sqrt(0.5) 
    phase = jax.random.uniform(k1, minval=0, maxval=2*jnp.pi)
    alpha_0 = radius * jnp.exp(1j * phase)
    phi_hom = (1/jnp.sqrt(3))*2.0*jnp.real(alpha_0 * jnp.exp(-(1j*omega_0 + 0.5*kappa) * t_grid))
    
    transfer_matrix = setup_noise_parameters(t_grid, omega_0, kappa, kBT)
    phi_noise_traj = generate_noise_fast(k2, transfer_matrix)[:, 0] 
    
    # We apply the scaling here: lambda_zpe * (2g)
    xi_total_x = 2.0*g*(phi_hom + phi_noise_traj)
    
    return jnp.zeros((t_grid.shape[0], 3)).at[:, 0].set(xi_total_x)


def compute_memory_kernel(tau_grid, omega_0, kappa, g=1.0):
    """
    Eq. (23) scaled by lambda_zpe**2 to maintain the FDT and correct the Lamb Shift.
    """
    # The original kernel has a factor of 4. We scale it by lambda_zpe squared.
    gamma_kernel = 4.0*jnp.exp(-0.5 * kappa * tau_grid) * jnp.sin(omega_0 * tau_grid)
    return (g**2)*gamma_kernel


@jax.jit
def generate_noise_fast(key, transfer_matrix):
    """
    Generates the stochastic noise trajectory Xi(t).
    
    In this model, the single boson exclusively couples to sigma_x, 
    so the total fluctuating noise Xi(t) only drives the x-component.
    """
    num_omega = transfer_matrix.shape[1]
    num_steps = transfer_matrix.shape[0]
    
    # We only need noise for the X-component (column 0) 
    # as per Eq. (22): Xi(t) = -2g * phi_fluct(t) * x_hat
    key_a, key_b = jax.random.split(key)
    
    # Sampling standard normal variables for the spectral decomposition
    noise_A = jax.random.normal(key_a, (num_omega,))
    noise_B = jax.random.normal(key_b, (num_omega,))
    
    Z_stoch = noise_A + 1j * noise_B

    # Project frequencies into the time domain
    # result shape: (num_steps,)
    xi_x = jnp.real(transfer_matrix @ Z_stoch)
    
    # Construct 3D noise vector: [Xi_x(t), 0, 0]
    # Based on Eq. (22) in your notes
    xi_t = jnp.zeros((num_steps, 3)).at[:, 0].set(xi_x)
    
    return xi_t


def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, 
                            noise_traj, B_field, dt):
    """
    Total effective field in the Lab Frame.
    """
    # 1. Memory Integral (Dissipation)
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices
    
    # Causal kernel slice
    gamma_causal = jnp.where(lag_indices > 0, 
                             jnp.take(gamma_kernel, lag_indices, mode='fill', fill_value=0.0), 
                             0.0)

    # Standard dissipative back-action
    memory_x = -1.0 * (jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    
    # 2. Fluctuating Field (Noise)
    xi_t = noise_traj[jnp.clip(step_idx, 0, noise_traj.shape[0]-1)]
    
    # 3. Final Field
    eff_field_x = B_field[0] + xi_t[0] + memory_x
    
    return jnp.array([eff_field_x, B_field[1], B_field[2]])


@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    """
    Corrected geometry for dS/dt = B_eff x S
    """
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]
    
    B_eff_curr = compute_effective_field(
        S_curr, state_trajectory, curr_idx, 
        gamma_kernel, noise_traj, B_field, dt
    )
    
    # FIXED: The predictor must follow B x S, not S x B
    S_pred = S_curr + jnp.cross(B_eff_curr, S_curr) * dt
    
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    
    B_eff_next = compute_effective_field(
        S_pred, traj_with_pred, step_idx, 
        gamma_kernel, noise_traj, B_field, dt
    )
    
    B_mid = 0.5 * (B_eff_curr + B_eff_next)
    
    b_norm = jnp.linalg.norm(B_mid) + 1e-12
    k = B_mid / b_norm
    theta = b_norm * dt
    
    # Rotation formula correctly applies k x S
    k_cross_S = jnp.cross(k, S_curr)
    k_dot_S = jnp.dot(k, S_curr)
    
    S_next = (S_curr * jnp.cos(theta) + 
              k_cross_S * jnp.sin(theta) + 
              k * k_dot_S * (1.0 - jnp.cos(theta)))
    
    new_state_traj = state_trajectory.at[step_idx].set(S_next)
    return new_state_traj, S_next


def run_twa_bundle(keys, t_grid, omega_0, kappa, kBT, B_field, g, initial_direction, batch_size=1000):
    """
    Manages the DTWA simulation by parallelizing trajectories across JAX devices.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # 1. Pre-compute Kernel (Analytical form from your Eq. 23)
    # This captures the non-Markovian memory of the single boson
    gamma_kernel_fine = compute_memory_kernel(t_grid, omega_0, kappa, g)
    
    # 3. Define the Single Trajectory Solver
    def solve_single_trajectory(key):
        # We need 3 keys now: Spin initial state, Cavity initial state, Bath noise
        k_samp, k_hom, k_bath = jax.random.split(key, 3)
        
        # A. Sample Initial State (Discrete Spin Sampling)
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction, coupling_type='full')
        
        # B. Generate Noise (Xi(t) restricted to the x-axis)
        # 1. Transient noise from the initial state of the cavity
        hom_noise_traj = generate_complete_noise(k_hom, t_grid, omega_0, kappa, g, kBT)
        
        # 2. Continuous vacuum/thermal fluctuations from the bath
        #bath_noise_traj = generate_noise_fast(k_bath, noise_transfer_matrix)
        
        # 3. Total fluctuating noise guarantees FDT is satisfied at all times
        noise_traj = hom_noise_traj 
        
        # C. Initialize History Array
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        # D. Time Loop using jax.lax.scan
        def scan_body(carry, idx):
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, B_field, dt)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    # 4. Batch Processor with JAX vmap
    @jax.jit
    def process_batch_sum(batch_keys):
        batch_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_trajs, axis=0)

    # 5. Main Averaging Loop
    total_sum = jnp.zeros((num_steps, 3))
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"3. Starting DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="DTWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_partial_sum = process_batch_sum(current_keys)
        total_sum = total_sum + batch_partial_sum
        
    # 6. Final DTWA Average Result
    return total_sum / n_total