import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Enable 64-bit precision for better numerical stability in physics simulations
jax.config.update("jax_enable_x64", True)

def get_spectral_density(omega, eta, omega_c, s):
    """
    Computes the spectral density J(omega) for the Ohmic family.
    
    Formula: J(omega) = eta * omega * (omega/omega_c)^(s-1) * exp(-omega/omega_c)
    
    Parameters:
    -----------
    omega : jnp.array
        Input frequencies (must be > 0).
    eta : float
        Coupling strength.
    omega_c : float
        Cutoff frequency.
    s : float
        Ohmicity exponent.
        s = 1 : Ohmic
        s < 1 : Sub-Ohmic (e.g., 0.5)
        s > 1 : Super-Ohmic (e.g., 3.0)
        
    Returns:
    --------
    jnp.array
        The spectral density J(omega).
    """
    # We use jnp.where to ensure J(omega) = 0 for omega <= 0 to avoid NaNs with fractional powers
    
    # Calculate the power law part safely
    # (omega / omega_c)^(s-1)
    # Note: If omega is 0 and s < 1, this technically diverges, 
    # but J(omega) ~ omega^s, so as long as s > 0, the total function is 0 at 0.
    
    # safe_omega avoids division by zero or log(0) issues during power calculation
    safe_omega = jnp.where(omega > 0, omega, 1.0) 
    
    prefactor = eta * safe_omega
    power_term = jnp.power(safe_omega / omega_c, s - 1)
    cutoff_term = jnp.exp(-safe_omega / omega_c)
    
    J_val = prefactor * power_term * cutoff_term
    
    # Force 0 where omega <= 0
    return jnp.where(omega > 0, J_val, 0.0)


def compute_correlation_function(times, spectral_density_fn, beta, w_max, n_steps):
    """
    Computes the correlation function C(t) using a Riemann sum integration.
    
    Parameters:
    -----------
    times : jnp.array
        Array of time points t to evaluate C(t).
    spectral_density_fn : function
        A function J(omega) that returns the spectral density array.
    beta : float
        Inverse temperature (1/kT).
    w_max : float
        Maximum frequency for integration limit.
    n_steps : int
        Number of integration steps (Riemann grid size).
        
    Returns:
    --------
    C_t : jnp.array (complex)
        The complex correlation function C(t).
    """
    
    # 1. Discretize Frequency Domain
    # We start slightly above 0 to avoid division by zero in coth(beta*w/2)
    # The contribution at exactly w=0 is usually 0 for Ohmic baths anyway.
    d_omega = w_max / n_steps
    omegas = jnp.linspace(d_omega, w_max, n_steps)
    
    # 2. Precompute Frequency-Dependent Terms (Vectorized)
    # Get J(omega) values
    J_vals = spectral_density_fn(omegas)
    
    # Calculate the Thermal Factor: coth(beta * omega / 2)
    # coth(x) = 1 / tanh(x)
    coth_term = 1.0 / jnp.tanh(beta * omegas / 2.0)
    
    # The integrand prefactor: J(w) * coth(...) for Real part, J(w) for Imag part
    real_prefactor = J_vals * coth_term
    imag_prefactor = J_vals
    
    # 3. Define the Single-Step Integration Function
    # This function calculates C(t) for a *single* time point t
    def integrate_single_time(t):
        # Calculate the oscillating terms for all frequencies at once
        cos_vals = jnp.cos(omegas * t)
        sin_vals = jnp.sin(omegas * t)
        
        # Construct the integrand array
        # Real part: J(w) * coth(bw/2) * cos(wt)
        integrand_real = real_prefactor * cos_vals
        
        # Imaginary part: - J(w) * sin(wt)
        integrand_imag = -imag_prefactor * sin_vals
        
        # Sum over all frequencies (Riemann Sum) and multiply by d_omega
        integral = jnp.sum(integrand_real + 1j * integrand_imag) * d_omega
        return integral

    # 4. Vectorize over Time
    # vmap transforms the function that takes 'float t' into one that takes 'array t'
    C_t = jax.vmap(integrate_single_time)(times)
    
    return C_t