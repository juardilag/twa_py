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