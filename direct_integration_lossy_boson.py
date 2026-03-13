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