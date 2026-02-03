import jax.numpy as jnp
from jax import vmap

def get_spectral_density(omega, eta, omega_c, s):
    # J(omega) = eta * omega^s * omega_c^(1-s) * exp(-omega/omega_c)
    safe_omega = jnp.where(omega > 0, omega, 1.0) 
    
    # Using jnp.power once is more efficient
    J_val = eta * safe_omega * jnp.power(safe_omega / omega_c, s - 1) * jnp.exp(-safe_omega / omega_c)
    
    return jnp.where(omega > 0, J_val, 0.0)


def get_A_omega(omega, eta, omega_c, s):
    """Spectral function A(omega) = 2 * J(omega) for positive omega."""
    return 2.0 * get_spectral_density(omega, eta, omega_c, s)


def compute_memory_kernel(tau_grid, eta, omega_c, s, g=1.0, num_omega=10_000):
    """
    Computes gamma(tau) = g^2 * integral(A(omega)/pi * sin(omega*tau) d_omega)
    """
    # Define an integration grid for omega
    # We go up to 10*omega_c to ensure the exponential cutoff has acted
    omega_max = 100*omega_c
    omega_grid = jnp.linspace(0, omega_max, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    A_vals = get_A_omega(omega_grid, eta, omega_c, s)
    
    def single_tau_kernel(t):
        # The integral from your notes
        integrand = (A_vals / jnp.pi) * jnp.sin(omega_grid * t)
        return g**2 * jnp.sum(integrand)*d_omega

    # Vmap over the time grid to get the kernel series
    return vmap(single_tau_kernel)(tau_grid)