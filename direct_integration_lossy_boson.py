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
    
    gamma_kernel = 4.0 * jnp.exp(-0.5 * kappa * tau_grid) * jnp.sin(omega_0 * tau_grid)
    
    return (g**2) * gamma_kernel


def setup_noise_parameters(t_grid, omega_0, kappa, kBT, g=1.0):
    """
    Sets up the noise transfer matrix for the single lossy boson.
    Satisfies FDT based on the boson Green's function poles.
    """
    # In the single-mode case, we don't need a massive omega_grid.
    # We sample frequencies around omega_0 to reconstruct the damped noise.
    # To resolve the width kappa, we need a grid covering several widths.
    num_omega = 2000 
    omega_grid = jnp.linspace(omega_0 - 5*kappa, omega_0 + 5*kappa, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    # The effective spectral density for a single lossy boson is a Lorentzian:
    # J_eff(w) = (kappa / 2π) / ((w - w0)^2 + (kappa/2)^2)
    lorentzian = (kappa / (2.0 * jnp.pi)) / ((omega_grid - omega_0)**2 + (kappa/2.0)**2)
    
    # Quantum Thermal Factor (coth) for the FDT [cite: 43]
    safe_omega = jnp.maximum(omega_grid, 1e-12)
    x = safe_omega / (2.0 * kBT)
    thermal_factor = 1.0 / jnp.tanh(x)
    
    # Amplitude coefficient handles the coupling g and the FDT 
    # We use 2.0 * g because Xi(t) = -2g * phi_fluct(t) 
    amplitude = 2.0 * g * jnp.sqrt(lorentzian * thermal_factor * d_omega)
    
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
    Computes the effective magnetic field vector as defined in Eq. (17) and (21).
    B_eff = B_ext + Xi(t) + Dissipative Memory
    """
    # 1. Vectorized Memory Convolution (X-component only)
    # Based on Eq. (23), the memory only acts on the x-component
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices

    # Fetch Gamma values (Memory Kernel)
    gamma_vals = jnp.take(gamma_kernel, lag_indices, mode='fill', fill_value=0.0)
    
    # Causality: mask out current and future time steps for the history integral
    gamma_causal = jnp.where(lag_indices > 0, gamma_vals, 0.0)

    # Calculate the Memory Integral for the X-component only 
    # history_array[:, 0] is the x-component of the spin history
    memory_history_x = jnp.dot(gamma_causal, history_array[:, 0]) * dt
    
    # Instantaneous back-action at t' = t
    memory_instant_x = gamma_kernel[0] * S_state[0] * dt
    
    # Total Dissipative field (strictly longitudinal to the x-axis) 
    phi_memory_x = memory_history_x + memory_instant_x

    # 2. Add Stochastic Noise (Already restricted to X in previous step) [cite: 82]
    xi_t = noise_traj[jnp.clip(step_idx, 0, noise_traj.shape[0]-1)]
    
    # 3. Combine with External Magnetic Field [cite: 65, 80]
    # B_field = [Bx, By, Bz]
    # The total contribution from the boson (Noise + Memory) is only in X
    eff_field_contrib = xi_t.at[0].add(phi_memory_x) 

    return B_field + eff_field_contrib


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