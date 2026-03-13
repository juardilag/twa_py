import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm.auto import tqdm

def compute_memory_kernel(tau_grid, omega_0, kappa, g=1.0):
    """
    Computes the analytical memory kernel for the single lossy boson.
    Based on Eq. (23) from the provided notes.
    
    Args:
        tau_grid: Array of time lags (t - t')
        omega_0: Frequency of the single boson mode
        kappa: Loss rate of the boson into the continuum
        g: Coupling strength between spin and boson
    """
    # From Eq. (23): D^R(tau) = 4 * exp(-kappa/2 * tau) * sin(omega_0 * tau)
    # We multiply by g^2 as per Eq. (21) in the notes
    
    gamma_kernel = 4*jnp.exp(-0.5 * kappa * tau_grid) * jnp.sin(omega_0 * tau_grid)
    
    return (g**2)*gamma_kernel


def setup_noise_parameters(t_grid, omega_0, kappa, kBT, g=1.0):
    """
    Sets up the noise transfer matrix for the single lossy boson.
    Satisfies FDT based on the boson Green's function poles.
    """
    # In the single-mode case, we don't need a massive omega_grid.
    # We sample frequencies around omega_0 to reconstruct the damped noise.
    # To resolve the width kappa, we need a grid covering several widths.
    num_omega = 10_000 
    omega_grid = jnp.linspace(omega_0 - 20*kappa, omega_0 + 20*kappa, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    # The effective spectral density for a single lossy boson is a Lorentzian:
    # J_eff(w) = (kappa / 2π) / ((w - w0)^2 + (kappa/2)^2)
    lorentzian = (kappa / (2.0 * jnp.pi)) / ((omega_grid - omega_0)**2 + (kappa/2.0)**2)
    print(f"Lorentzian Area: {jnp.sum(lorentzian) * d_omega:.4f}")
    
    # Quantum Thermal Factor (coth) for the FDT [cite: 43]
    safe_omega = jnp.maximum(omega_grid, 1e-12)
    x = safe_omega / (2.0 * kBT)
    thermal_factor = 1.0 / jnp.tanh(x)
    
    # Amplitude coefficient handles the coupling g and the FDT 
    # We use 2.0 * g because Xi(t) = -2g * phi_fluct(t) 
    amplitude = 2*g*jnp.sqrt(lorentzian * thermal_factor * d_omega)
    
    # Time evolution matrix: exp(-i * w * t) 
    time_evolution = jnp.exp(-1j * jnp.outer(t_grid, omega_grid))
    transfer_matrix = time_evolution * amplitude[None, :]
    
    return transfer_matrix

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
    EOM built directly from QuTiP H = 0.5*B*sigma + g*sx*(a+adag)
    """
    # 1. Precompute the memory integral (Dissipation)
    # Note: gamma_kernel should be: 4 * exp(-kappa/2 * tau) * sin(omega_0 * tau)
    # The factor of 4 comes from (2g)^2 derivation in Eq. (23) 
    
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices
    gamma_causal = jnp.where(lag_indices > 0, 
                             jnp.take(gamma_kernel, lag_indices, mode='fill', fill_value=0.0), 
                             0.0)

    # memory_val = g^2 * integral[ D^R(t-t') * sigma_x(t') ]
    # We use a negative sign here because the back-action MUST be dissipative (friction)
    memory_x = -1.0 * (jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    
    # 2. Add Stochastic Noise Xi(t) = 2g * phi_fluct
    # Ensure setup_noise_parameters uses 'amplitude = 2.0 * g * ...'
    xi_t = noise_traj[jnp.clip(step_idx, 0, noise_traj.shape[0]-1)]
    
    # 3. Total Effective Field
    # Only the x-component is modified by the boson coupling [cite: 81, 85]
    eff_field_x = B_field[0] + xi_t[0] + memory_x
    
    return jnp.array([eff_field_x, B_field[1], B_field[2]])


@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    """
    Performs a single step of the Heun (Predictor-Corrector) integration.
    Adapted for the non-Markovian spin EOM from the provided notes.
    """
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]
    
    # 1. Predictor Step: Estimate S_pred at t + dt
    # Compute field based on current state and history
    B_eff_curr = compute_effective_field(
        S_curr, state_trajectory, curr_idx, 
        gamma_kernel, noise_traj, B_field, dt
    )
    
    # Standard Euler predictor
    S_pred = S_curr + jnp.cross(S_curr, B_eff_curr) * dt
    
    # Temporarily update the trajectory with the prediction to compute the next field
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    
    # 2. Corrector Step: Re-estimate field at t + dt using the prediction
    B_eff_next = compute_effective_field(
        S_pred, traj_with_pred, step_idx, 
        gamma_kernel, noise_traj, B_field, dt
    )
    
    # 3. Average the fields (Trapezoidal rule)
    B_mid = 0.5 * (B_eff_curr + B_eff_next)
    
    # 4. Geometric Rotation (Rodrigues' Rotation Formula)
    # This ensures the spin norm |S|=1 is preserved exactly.
    b_norm = jnp.linalg.norm(B_mid) + 1e-12
    k = B_mid / b_norm
    theta = b_norm * dt
    
    k_cross_S = jnp.cross(k, S_curr)
    k_dot_S = jnp.dot(k, S_curr)
    
    S_next = (S_curr * jnp.cos(theta) + 
              k_cross_S * jnp.sin(theta) + 
              k * k_dot_S * (1.0 - jnp.cos(theta)))
    
    # Update the permanent trajectory array
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
    print("1. Computing Analytical Memory Kernel...")
    gamma_kernel_fine = compute_memory_kernel(t_grid, omega_0, kappa, g)
    
    # 2. Pre-compute Noise Transfer Matrix
    # This satisfies the FDT for the Lorentzian spectral density of the lossy boson
    print("2. Pre-computing Noise Transfer Matrix...")
    noise_transfer_matrix = setup_noise_parameters(t_grid, omega_0, kappa, kBT, g)
    
    # 3. Define the Single Trajectory Solver
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        # A. Sample Initial State (Discrete Spin Sampling)
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction, coupling_type = 'full')
        
        # B. Generate Noise (Xi(t) restricted to the x-axis)
        noise_traj = generate_noise_fast(k_noise, noise_transfer_matrix)
        
        # C. Initialize History Array
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        # D. Time Loop using jax.lax.scan for high-performance execution
        def scan_body(carry, idx):
            # carry is the state_trajectory
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